# GAN Image Generation

This project implements a Generative Adversarial Network (GAN) using TensorFlow to generate images. The system includes the functionality to save the model, test its performance by generating images, and evaluate the quality and diversity of the generated images using several evaluation techniques. It serves as a practical, detailed, and well-structured implementation of GANs, useful for both understanding and applying GANs in image generation tasks.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [References](#references)

## Overview

Generative Adversarial Networks (GANs) are powerful models used to generate data, typically images, by learning the underlying distribution of a dataset. This project aims to:
- Build and train a GAN from scratch using TensorFlow.
- Evaluate the quality of generated images using techniques like discriminator accuracy, Frechet Inception Distance (FID), and image diversity.
- Visualize and interpret model performance, making the process of learning GANs intuitive and practical.

This project represents a complete, professional implementation of GANs, combining model creation, testing, and evaluation in a clear and organized way.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmdmohamedd/gan-image-generation.git
   cd gan-image-generation
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n gan-env python=3.8
   conda activate gan-env
   ```

3. Install the required packages:
   ```bash
   pip install tensorflow matplotlib
   ```

## Dataset

The GAN is trained on the **Fashion MNIST** dataset, which contains 70,000 grayscale images of 10 different fashion categories. Each image is 28x28 pixels in size.

To load the dataset in TensorFlow:

```python
(train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]
```

## Model Architecture

This project uses a standard GAN architecture, consisting of two components:

### 1. Generator
The generator is responsible for creating new data. It takes random noise as input and generates an image as output. The architecture includes:
- Dense layers
- Batch normalization
- Leaky ReLU activations
- Up-sampling using transposed convolution layers

### 2. Discriminator
The discriminator evaluates whether an image is real or generated. It consists of:
- Convolutional layers
- Leaky ReLU activations
- Dropout for regularization
- Outputs a probability (real or fake)

## Training

Training involves alternating between the generator and discriminator:
1. Train the discriminator on both real and generated (fake) images.
2. Train the generator through the feedback of the discriminator.

The loss functions used:
- **Generator Loss**: Binary crossentropy on the discriminator's predictions of fake images.
- **Discriminator Loss**: Binary crossentropy on real and fake images.

Training is done for a number of epochs (default: 50), and the models are saved after each epoch.

## Evaluation

### 1. **Discriminator Accuracy**
Evaluates the performance of the discriminator in distinguishing between real and generated images.

### 2. **Frechet Inception Distance (FID)**
A standard measure to evaluate the quality of generated images by comparing the statistics of real and generated image datasets.

### 3. **Image Diversity**
Evaluates the diversity of generated images to ensure the generator is not simply replicating a single pattern.

### 4. **Loss Curves**
Track the loss of both the generator and discriminator during training to monitor the learning process.

## Results

### Discriminator Accuracy
- **Real Image Accuracy**: 72%
- **Fake Image Accuracy**: 54%

These values indicate that the discriminator is fairly good at distinguishing real images but still challenged by generated ones, implying room for improvement in the generator.

### Sample Generated Images
Generated images from the GAN after training can be visualized to assess their quality and diversity.

![Sample Generated Images](path/to/sample_image.png)

## Usage

### Train the GAN
To train the GAN, simply run the notebook or execute the following command after configuring your environment:

```bash
jupyter notebook gan_image_generation.ipynb
```

### Test the Generator
You can test the generator by loading a saved model and generating images:

```python
test_generator('generator_epoch_50.h5')
```

### Evaluate the Model
Use evaluation metrics to assess the performance of the GAN:

```python
calculate_discriminator_accuracy(real_images, fake_images)
calculate_fid('/path/to/real_images', '/path/to/generated_images')
```

## References

- Ian Goodfellow et al., "Generative Adversarial Networks", 2014.
- Fashion MNIST Dataset: [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- Frechet Inception Distance: [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)
