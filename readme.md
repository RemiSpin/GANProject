# Teaching AI to Paint Like Monet: A GAN Implementation

## What is a GAN and How Does It Work?

**Generative Adversarial Networks (GANs)** are a machine learning architecture consisting of two neural networks competing against each other in a zero-sum game.

**The Core Concept:**
GANs operate like an art forger competing against an art expert. The forger creates fake paintings while the expert tries to identify forgeries. Through this competition, both improve their skills until the forger creates works indistinguishable from authentic pieces.

**The Two Components:**

**Generator (The Artist):** Takes random noise as input and transforms it into realistic images. Initially produces poor quality outputs that gradually improve through training.

**Discriminator (The Critic):** Analyzes images and classifies them as real (from training data) or fake (from generator). Provides feedback that guides the generator's improvement.

**Training Process:** Both networks train simultaneously. The generator learns to create increasingly convincing fakes, while the discriminator becomes better at detection. This adversarial process drives both networks toward optimal performance.

**Why GANs Work:** The discriminator provides continuous, specific feedback about what makes images look "real" or "fake." This guidance is far more effective than training a generator in isolation without quality assessment.

---

## Monet Style

**Project Objective:** Create an AI system capable of generating new paintings in Claude Monet's impressionist style.

**Dataset:** 500 authentic Monet paintings from the Kaggle "I'm Something of a Painter Myself" competition, preprocessed and normalized for neural network training.

**Architecture Choices:**

- **Generator:** U-Net architecture with skip connections to preserve fine details while learning global style patterns
- **Discriminator:** PatchGAN design that analyzes local image patches rather than entire images, focusing on texture and brushstroke authenticity

---

## Technical Details

**Advanced Noise Generation:** Implemented multi-modal noise mixing (normal, uniform, and Laplace distributions) with time-varying weights to ensure output diversity and prevent mode collapse.

**Quality Control Systems:** Automated assessment of color variance and image diversity to detect training issues and implement early stopping when quality thresholds are met.

**U-Net Skip Connections:** Generator architecture preserves fine details by concatenating early layer features with later layers, allowing both global style learning and detail preservation.

**Spectral Normalization:** Applied to discriminator layers to prevent training instability and ensure balanced competition between generator and discriminator.

**Data Augmentation:** Smart horizontal and vertical flipping with controlled probability to increase dataset diversity without overfitting to specific orientations.

**Label Smoothing:** Used soft labels (0.9/0.1 instead of 1.0/0.0) to prevent discriminator overconfidence and maintain healthy adversarial competition.

**Diversity Monitoring:** Pair-wise image comparison system to detect mode collapse and ensure varied output generation throughout training.
