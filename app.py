import os
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


st.set_page_config(page_title="cGAN Digit Generator", layout="wide")

st.title("🔢 cGAN - Conditional Digit Generator")
st.markdown("---")


NOISE_DIM = 100
WEIGHTS_PATH = "conditional_generator.weights.h5"



@st.cache_resource
def build_conditional_generator():
    noise_input = layers.Input(shape=(NOISE_DIM,), name="noise_input")

    label_input = layers.Input(shape=(1,), dtype=tf.int32, name="label_input")

    label_embedding = layers.Embedding(10, 50)(label_input)
    label_embedding = layers.Flatten()(label_embedding)

    combined = layers.Concatenate()([noise_input, label_embedding])

    x = layers.Dense(7 * 7 * 256, use_bias=False)(combined)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((7, 7, 256))(x)

    x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    output = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh")(x)

    model = keras.Model(inputs=[noise_input, label_input], outputs=output)
    return model



generator = build_conditional_generator()


if os.path.exists(WEIGHTS_PATH):

    generator.load_weights(WEIGHTS_PATH)
    weights_status = "✅ Trained cGAN weights loaded."
else:
    weights_status = "⚠️ Weights not found! Using random initialization."


st.sidebar.header("⚙️ Generation Settings")


target_digit = st.sidebar.selectbox(
    "Which digit should I draw?",
    options=list(range(10)),
    index=5  # Default to 5
)


num_images = st.sidebar.slider(
    "How many variations?",
    min_value=1,
    max_value=16,
    value=4,
    step=1,
)

st.sidebar.divider()
st.sidebar.caption(weights_status)


if st.sidebar.button("🚀 Generate Digit", use_container_width=True):
    st.info(f"Generating {num_images} variations of the digit **{target_digit}**...")


    noise = tf.random.normal([num_images, NOISE_DIM])

    labels = tf.constant([[target_digit]] * num_images)


    generated_images = generator([noise, labels], training=False)


    cols = min(num_images, 4)
    rows = int(np.ceil(num_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))


    if num_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(len(axes)):
        axes[i].axis("off")
        if i < num_images:

            img = generated_images[i].numpy().reshape(28, 28)
            img = (img * 0.5) + 0.5
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(f"Class: {target_digit}", fontsize=10)


    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    st.success("Successfully generated!")


st.markdown("---")
st.subheader("📊 System Architecture")

c1, c2, c3 = st.columns(3)
with c1:
    st.info("**Model Type**\n\nConditional GAN")
with c2:
    st.info("**Input Mode**\n\nNoise + Label Embedding")
with c3:
    st.info("**Latency**\n\n~15ms / image")

st.markdown("---")
st.caption("Powered by Streamlit, TensorFlow, and your trained cGAN weights.")