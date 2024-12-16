import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_shape):
    # Encoder
    input_img = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    encoded = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    return models.Model(input_img, decoded)



def build_cnn_reconstructor(input_shape):
    """
    CNN-based model for fingerprint reconstruction.
    It processes input images through multiple convolutional layers
    to directly reconstruct a cleaner version of the input.
    """
    input_img = layers.Input(shape=input_shape)

    # Convolutional Layers - Feature Extraction
    x = input_img
    # x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    # x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    # x = layers.MaxPooling2D((2, 2), padding='same')(x)  # Downsample 1
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # Downsample 2

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # Downsample 3
    
    # Upsampling Layers - Reconstruction
    
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # Upsample 1

    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # Upsample 2

    # x = layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
    # x = layers.UpSampling2D((2, 2))(x)  # Upsample 3
    
    output_img = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # Reconstructed image

    return models.Model(input_img, output_img)