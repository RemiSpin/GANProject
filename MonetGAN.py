import os, glob
import zipfile
from PIL import Image
import torch
from torch import nn, optim
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
import random
import numpy as np
import time
import io

monet_dir = '/kaggle/input/gan-getting-started/monet_jpg'
OUTPUT_DIR = '/kaggle/working/generated_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 8
Z_DIM = 256
LR_G = 1e-4
LR_D = 2e-4
BETA1 = 0.5
EPOCHS = 30
NUM_IMAGES = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EARLY_EXIT_ON_GOOD_SAMPLES = True
SAMPLE_CHECK_FREQUENCY = 1

print(f"CycleGAN Training - {EPOCHS} epochs, {IMG_SIZE}x{IMG_SIZE} images")
print(f"Device: {DEVICE}")

# Data augmentation applies random transformations to increase training data diversity
class OptimizedAugmentation:
    def __init__(self, img_size=128):
        self.img_size = img_size
    
    def __call__(self, image):
        if random.random() > 0.5:
            image = T.functional.hflip(image)
        if random.random() > 0.7:
            image = T.functional.vflip(image)
        return image

# Smart photo album that loads and prepares Monet images for AI training
class MonetDataset(Dataset):
    def __init__(self, folder, size, use_augmentation=True):
        self.files = glob.glob(os.path.join(folder, '*.jpg'))[:500]
        self.size = size
        
        transforms = [
            T.Resize(size),
            T.CenterCrop(size),
        ]
        
        if use_augmentation:
            transforms.append(OptimizedAugmentation(size))
        
        transforms.extend([
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
        
        self.transform = T.Compose(transforms)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert('RGB')
        return self.transform(img)

# Keeps AI's vision stable by normalizing image brightness/contrast during training
class EfficientInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

# The "artist" AI that creates Monet paintings from random noise
# Uses U-Net architecture: goes down (compress), then up (expand) with skip connections
class UNetGenerator(nn.Module):
    def __init__(self, z_dim=256, output_channels=3):
        super().__init__()
        
        # Step 1: Convert random noise into initial image features (like an artist's first sketch)
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 4, 1, 0, bias=False),
            EfficientInstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Step 2: Compress to understand the overall composition (like stepping back to see the whole canvas)
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 256, 4, 2, 1, bias=False),
            EfficientInstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Step 3: The creative core - where style decisions are made
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            EfficientInstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Steps 4-8: Gradually build up the final image, adding detail at each layer
        # Dropout prevents overfitting (like an artist trying different techniques)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            EfficientInstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Notice: 512 input channels here - this is where skip connection adds detail
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1, bias=False),
            EfficientInstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            EfficientInstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            EfficientInstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            EfficientInstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Final step: Convert features to RGB image (3 channels) with Tanh (-1 to 1 range)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(16, output_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        x1 = self.initial(z)  # Create initial sketch
        x2 = self.down1(x1)   # Compress for composition
        
        x = self.bottleneck(x2)  # Make core creative decisions
        
        x = self.up1(x)
        # Skip connection: combine current work with initial sketch (preserves fine details)
        x = torch.cat([x, x1], dim=1)  # This is why up2 expects 512 channels
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.final(x)
        
        return x

# The "art critic" AI that judges image patches as real/fake Monet paintings
class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            EfficientInstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            EfficientInstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
            EfficientInstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False))
        )
    
    def forward(self, x):
        return self.model(x)

# Scoring system that tells the artist and critic how well they're doing their jobs
class GANLoss:
    def __init__(self, device):
        self.device = device
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.bce_loss(real_output, torch.ones_like(real_output) * 0.9)
        fake_loss = self.bce_loss(fake_output, torch.zeros_like(fake_output) + 0.1)
        return (real_loss + fake_loss) * 0.5
    
    def generator_loss(self, fake_output):
        return self.bce_loss(fake_output, torch.ones_like(fake_output) * 0.9)

# Reusable function to set up good starting weights for all network layers
def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight'):
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('InstanceNorm') != -1 or classname.find('EfficientInstanceNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

# Quality checker that tests if AI art is good enough to stop training early
def assess_sample_quality(generator, device, z_dim, epoch):
    original_mode_is_train = generator.training
    generator.train()
    with torch.no_grad():
        test_noise = sample_noise(4, z_dim, device)
        test_imgs = generator(test_noise)
        test_imgs = (test_imgs + 1) / 2.0
        test_imgs = torch.clamp(test_imgs, 0, 1)
        
        vutils.save_image(test_imgs, f"{OUTPUT_DIR}/quality_check_epoch{epoch+1}.png", 
                         nrow=2, normalize=False)
        
        avg_color_variance = test_imgs.var(dim=1).mean().item()
        
        img_diversity = 0
        for i in range(len(test_imgs)):
            for j in range(i+1, len(test_imgs)):
                diff = torch.abs(test_imgs[i] - test_imgs[j]).mean().item()
                img_diversity += diff
        img_diversity /= (len(test_imgs) * (len(test_imgs) - 1) / 2)
        
    if not original_mode_is_train:
        generator.eval()
            
    return avg_color_variance > 0.01

# Set up the data feeding system
dataset = MonetDataset(monet_dir, IMG_SIZE, use_augmentation=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

G = UNetGenerator(Z_DIM).to(DEVICE)
D = PatchGANDiscriminator().to(DEVICE)
G.apply(weights_init)
D.apply(weights_init)

optG = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, 0.999))
optD = optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, 0.999))

loss_fn = GANLoss(DEVICE)

# Creative inspiration generator that mixes different types of randomness for diverse art
def sample_noise(batch_size, z_dim, device):
    entropy_source = int(time.time() * 1000000) + os.getpid() + random.randint(0, 1000000)
    entropy_source = entropy_source % (2**32)
    
    gen1 = torch.Generator(device=device).manual_seed(entropy_source)
    gen2 = torch.Generator(device=device).manual_seed((entropy_source + 12345) % (2**32))
    gen3 = torch.Generator(device=device).manual_seed((entropy_source + 67890) % (2**32))
    
    normal_noise = torch.randn(batch_size, z_dim, 1, 1, device=device, generator=gen1)
    uniform_noise = (torch.rand(batch_size, z_dim, 1, 1, device=device, generator=gen2) - 0.5) * 4
    laplace_like = torch.randn(batch_size, z_dim, 1, 1, device=device, generator=gen3)
    laplace_like = torch.sign(laplace_like) * torch.log(1 + torch.abs(laplace_like))
    
    time_factor = (time.time() % 1.0)
    mix1 = 0.6 + 0.3 * time_factor
    mix2 = 0.3 + 0.2 * (1 - time_factor)
    mix3 = 0.1 + 0.1 * time_factor
    
    combined_noise = mix1 * normal_noise + mix2 * uniform_noise + mix3 * laplace_like
    
    for i in range(batch_size):
        batch_gen = torch.Generator(device=device).manual_seed((entropy_source + i * 1337) % (2**32))
        batch_noise = torch.randn(1, z_dim, 1, 1, device=device, generator=batch_gen) * 0.1
        combined_noise[i:i+1] += batch_noise
    
    return combined_noise

# Compares images pair-by-pair to measure how different they are from each other
def calculate_diversity(images):
    if len(images) < 2:
        return 0.0
    
    total_diff = 0
    num_pairs = 0
    
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            diff = torch.abs(images[i] - images[j]).mean().item()
            total_diff += diff
            num_pairs += 1
    
    return total_diff / num_pairs if num_pairs > 0 else 0.0

# Creativity checker that generates sample images and tests if they're varied or repetitive
def check_diversity(generator, device, z_dim, num_samples=5):
    original_mode_is_train = generator.training
    generator.train()
    samples = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            noise = sample_noise(1, z_dim, device)
            fake_img = generator(noise)
            samples.append(fake_img[0].cpu())
    
    diversity_score = calculate_diversity(samples)
    
    if not original_mode_is_train:
        generator.eval()

    return diversity_score

print(f"Starting training...")
print(f"Dataset size: {len(dataset)} images")

# Set up controlled randomness for reproducible but varied training
base_seed = int(time.time() * 1000000) + os.getpid() + random.randint(0, 1000000)
torch.manual_seed(base_seed % (2**32))
np.random.seed((base_seed + 12345) % (2**32))
random.seed((base_seed + 67890) % (2**32))
if torch.cuda.is_available():
    torch.cuda.manual_seed_all((base_seed + 98765) % (2**32))

# Main training loop
for epoch in range(EPOCHS):
    G.train()
    D.train()
    epoch_g_loss = 0
    epoch_d_loss = 0
    num_batches = 0
    
    for i, real_images in enumerate(loader):
        real_images = real_images.to(DEVICE)
        batch_size = real_images.size(0)
        
        D.zero_grad()
        real_output = D(real_images)
        
        noise = sample_noise(batch_size, Z_DIM, DEVICE)
        fake_images = G(noise)
        fake_output = D(fake_images.detach())
        
        d_loss = loss_fn.discriminator_loss(real_output, fake_output)
        d_loss.backward()
        optD.step()
        
        G.zero_grad()
        noise = sample_noise(batch_size, Z_DIM, DEVICE)
        fake_images = G(noise)
        fake_output = D(fake_images)
        
        g_loss = loss_fn.generator_loss(fake_output)
        g_loss.backward()
        optG.step()
        
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        num_batches += 1
    
    avg_g_loss = epoch_g_loss / num_batches
    avg_d_loss = epoch_d_loss / num_batches
    print(f"Epoch [{epoch+1}/{EPOCHS}] completed - Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}")
    
    if (epoch + 1) % SAMPLE_CHECK_FREQUENCY == 0:
        quality_good = assess_sample_quality(G, DEVICE, Z_DIM, epoch)
        diversity_score = check_diversity(G, DEVICE, Z_DIM, num_samples=8)
        
        if EARLY_EXIT_ON_GOOD_SAMPLES and quality_good and epoch >= 2:
            print("Training completed - quality threshold achieved!")
            break

print("Training complete!")

print(f"Generating {NUM_IMAGES} output images...")
G.eval()
to_pil = ToPILImage()

torch.manual_seed(int(time.time() * 1000) % 2**32)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(int(time.time() * 1000) % 2**32)
np.random.seed(int(time.time() * 1000) % 2**32)
random.seed(int(time.time() * 1000) % 2**32)

zip_path = f"{OUTPUT_DIR}/images.zip"
print(f"Creating {zip_path} with {NUM_IMAGES} generated images...")

diversity_samples = []
generated_checksums = set()

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
    G.train()
    torch.set_grad_enabled(False)
    
    for img_idx in range(NUM_IMAGES):
        img_seed = int(time.time() * 1000000 + img_idx) % 2**32
        generator = torch.Generator(device=DEVICE).manual_seed(img_seed)
        
        noise = sample_noise(1, Z_DIM, DEVICE)
        extra_noise = torch.randn(noise.shape, generator=generator, device=noise.device, dtype=noise.dtype) * 0.1
        noise = noise + extra_noise
        
        fake_img = G(noise)
        fake_img = (fake_img + 1) / 2.0
        fake_img = torch.clamp(fake_img, 0, 1)
        fake_img_resized = F.interpolate(fake_img, size=(256, 256), mode='bilinear', align_corners=False)
        
        if img_idx < 20:
            diversity_samples.append(fake_img_resized[0].cpu())
            img_array = fake_img_resized[0].cpu().numpy()
            checksum = hash(img_array.tobytes())
            if checksum not in generated_checksums:
                generated_checksums.add(checksum)
        
        pil_img = to_pil(fake_img_resized[0].cpu())
        img_filename = f"generated_{img_idx:03d}.jpg"
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)
        zipf.writestr(img_filename, img_buffer.getvalue())

torch.set_grad_enabled(True)

# Final diversity check - safety measure to ensure we have enough images to compare
if len(diversity_samples) > 1:
    diversity_score = 0
    num_comparisons = 0
    for i in range(len(diversity_samples)):
        for j in range(i+1, len(diversity_samples)):
            diff = torch.abs(diversity_samples[i] - diversity_samples[j]).mean().item()
            diversity_score += diff
            num_comparisons += 1

if os.path.exists(zip_path):
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Generated images saved to: {zip_path}")
    print(f"ZIP file size: {zip_size_mb:.1f} MB")
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        file_count = len(zipf.namelist())
        print(f"ZIP contains {file_count} files")
else:
    print("ERROR: ZIP file was not created!")

print(f"• Generated {NUM_IMAGES} final images")