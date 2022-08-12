#!/usr/bin/env python
# coding: utf-8

# # Read tiles to input format VAE network

# ### Imports
# Install tensorflow:
# ``%pip install tensorflow``

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
tf.random.set_seed(2) 


# ### Create sampling layer


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ### Build encoder

def make_encoder(cutout_size,n_bands,
                 filter_1,filter_2,
                 kernel_size_1,kernel_size_2,
                 dense_size,latent_dim):    
    encoder_inputs = keras.Input(shape=(cutout_size, cutout_size,n_bands)) # enter cut-out shape (20,20,3)
    x = layers.Conv2D(filter_1, kernel_size_1, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(filter_2, kernel_size_2, activation="relu", strides=2, padding="same")(x)
    # add a third layer (not sure if that makes sense)
    x = layers.Conv2D(filter_2, kernel_size_2, activation="relu", strides=1, padding="same")(x)
    x = layers.Flatten()(x) # to vector
    x = layers.Dense(dense_size, activation="relu")(x) # linked layer
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder_inputs, encoder, z , z_mean, z_log_var


# ### Build decoder

def make_decoder(latent_dim,encoder,
                 filter_1,filter_2,
                 kernel_size_1,kernel_size_2,
                 n_bands): 
    latent_inputs = keras.Input(shape=(latent_dim,))
    # get shape of last layer in encoder before flattning
    flat_layer = [layer for layer in encoder.layers if 'flatten' in layer.name] 
    flat_input = flat_layer[-1].input_shape # input shape of flat layer to be used to reconstruct; (None, 5,5,16) or smth
    x = layers.Dense(flat_input[1] * flat_input[2] * filter_2, activation="relu")(latent_inputs) # -- shape corresponding to encoder
    x = layers.Reshape((flat_input[1], flat_input[2], filter_2))(x)
    x = layers.Conv2DTranspose(filter_2, kernel_size_2, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(filter_2, kernel_size_2, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(filter_1, kernel_size_1, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(n_bands, n_bands, activation="sigmoid", padding="same")(x) # (1,3) or (3,3)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder


# ## Define VAE as model
# With custom train_step

# Update: instead of defining VAE as class, use function-wise definition

# Define VAE model.
def make_vae(encoder_inputs, z, z_mean, z_log_var, decoder,alpha=5):
    outputs = decoder(z)
    vae = tf.keras.Model(inputs=encoder_inputs, outputs=outputs, name="vae")

    # Add KL divergence regularization loss.
    reconstruction = decoder(z)
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            keras.losses.binary_crossentropy(encoder_inputs, reconstruction), axis=(1, 2)
                )
            )
    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    
    total_loss = reconstruction_loss +  alpha * kl_loss # alpha is custom
    vae.add_loss(total_loss)
    
#     kl_loss = alpha * kl_loss
    
#     vae.add_loss(reconstruction_loss)
#     vae.add_loss(kl_loss)

#     vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
#     vae.add_metric(reconstruction_loss, name='rconstr_loss', aggregation='mean')
    
    return vae
    

