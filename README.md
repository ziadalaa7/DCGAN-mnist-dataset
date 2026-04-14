# DCGAN for MNIST Digit Generation

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate handwritten digits using the classic **MNIST** dataset. It demonstrates the power of generative models in creating realistic synthetic data.

## 🚀 Project Overview
The goal of this project is to train two neural networks—a **Generator** and a **Discriminator**—in a competitive setting. The generator learns to create realistic images of digits, while the discriminator learns to distinguish between real images from the dataset and fake ones produced by the generator.

## 🛠️ Tech Stack
* **Framework:** TensorFlow / Keras
* **Architecture:** Convolutional Neural Networks (CNN)
* **Dataset:** MNIST (60,000 grayscale images of digits 0-9)
* **Optimization:** Adam Optimizer

## 🏗️ Model Architecture

### Generator
The generator takes a random noise vector (latent space) and transforms it into a 28x28 grayscale image.
* **Input:** Latent vector of size 100.
* **Layers:** Dense layer, followed by multiple **Conv2DTranspose** (Deconvolution) layers.
* **Normalization:** Batch Normalization is used to stabilize training.
* **Activation:** ReLU for hidden layers and **Tanh** for the output layer.

### Discriminator
The discriminator is a binary classifier that determines if a given image is real or fake.
* **Input:** 28x28x1 image.
* **Layers:** **Conv2D** layers with strided convolutions.
* **Activation:** LeakyReLU for hidden layers and **Sigmoid** for the final output.
* **Regularization:** Dropout layers to prevent the discriminator from overpowering the generator.

## 📁 File Structure
* `gans.ipynb`: The main Jupyter notebook containing data preprocessing, model definitions, training loops, and visualization of results.
* `requirements.txt`: List of necessary Python libraries to run the project.
* `README.md`: Project documentation.

## ⚙️ How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ziadalaa7/DCGAN-mnist-dataset.git](https://github.com/ziadalaa7/DCGAN-mnist-dataset.git)
