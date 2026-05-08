import os
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Page configuration
st.set_page_config(page_title="DCGAN Digit Generator", layout="wide")

st.title("🎨 DCGAN - Handwritten Digit Generator")
st.markdown("---")

# Constants
NOISE_DIM = 100
BATCH_SIZE = 128
WEIGHTS_PATH = "generator.weights.h5"

# Model builder
@st.cache_resource
def build_generator():
    """Build the DCGAN generator (unconditional)."""
    model = keras.Sequential(
        [
            layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(NOISE_DIM,)),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Reshape((7, 7, 256)),
            layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(
                1,
                (5, 5),
                strides=(2, 2),
                padding="same",
                use_bias=False,
                activation="tanh",
            ),
        ]
    )
    return model

# Build the generator
generator = build_generator()

if os.path.exists(WEIGHTS_PATH):
    generator.load_weights(WEIGHTS_PATH)
    weights_status = "Loaded trained weights."
else:
    weights_status = "Weights not found. Using random initialization."

st.sidebar.header("⚙️ Settings")

num_images = st.sidebar.slider(
    "Number of images to generate:",
    min_value=1,
    max_value=16,
    value=4,
    step=1,
)

st.sidebar.caption(weights_status)

if st.sidebar.button("🚀 Generate images", use_container_width=True):
    st.info("Generating images...")

    noise = tf.random.normal([num_images, NOISE_DIM])
    generated_images = generator(noise, training=False)

    cols = min(num_images, 4)
    rows = int(np.ceil(num_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))

    if num_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(len(axes)):
        axes[i].axis("off")
        if i < num_images:
            img = generated_images[i].numpy().reshape(28, 28)
            img = (img * 0.5) + 0.5
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title("Generated", fontsize=11, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)

    st.success("✅ Images generated successfully!")

# Model info
st.markdown("---")
st.subheader("📊 Model information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset", "MNIST (0-9)")

with col2:
    st.metric("Image size", "28×28")

with col3:
    st.metric("Noise vector", NOISE_DIM)

st.markdown("---")
st.markdown("Built with Streamlit and TensorFlow | DCGAN Model")
