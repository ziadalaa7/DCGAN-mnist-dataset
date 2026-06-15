# 🎨 Conditional DCGAN (CDCGAN) for MNIST Digit Generation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A Deep Convolutional Generative Adversarial Network extended with a conditional mechanism (CDCGAN). This model learns to generate highly realistic, specific handwritten digits (0-9) from the MNIST dataset by conditioning both the Generator and Discriminator on class labels.

## 🚀 Key Features
- **Controlled Generation:** Explicitly control the generated digit by providing a specific class label (0-9) along with the latent noise vector.
- **Label Embedding:** Maps discrete class labels into dense 50-dimensional vectors to capture semantic class information before concatenating them with the feature maps.
- **Advanced Architecture:** Replaces standard fully connected layers with Transposed Convolutions for upsampling and Strided Convolutions for feature extraction.
- **Stable Adversarial Training:** Utilizes Batch Normalization, LeakyReLU activations, and Dropout regularization to prevent mode collapse and ensure stable convergence.

## 🧠 Architecture Overview

### 1. Conditional Generator
- **Inputs:** 100-dimensional Gaussian noise + 1-dimensional class label.
- **Processing:** The label is embedded into a 50D space and concatenated with the noise. It is then projected via a Dense layer and reshaped into a `7x7x256` spatial tensor.
- **Upsampling:** Three progressive `Conv2DTranspose` layers upsample the feature maps to a final `28x28x1` image.
- **Activation:** `tanh` output layer to produce normalized pixel values [-1, 1].

### 2. Conditional Discriminator
- **Inputs:** `28x28x1` image + 1-dimensional class label.
- **Feature Extraction:** Extracts image features using two strided `Conv2D` layers.
- **Conditioning:** The spatial features are flattened and concatenated with the embedded class label to evaluate both image realism and label matching.
- **Classification:** Passes through a dense layer with LeakyReLU and 0.3 Dropout before the final binary classification using logits.

## ⚙️ Quick Start

**1. Clone the repository:**
```bash
git clone [https://github.com/ziadalaa7/DCGAN-mnist-dataset.git](https://github.com/ziadalaa7/DCGAN-mnist-dataset.git)
cd DCGAN-mnist-dataset
