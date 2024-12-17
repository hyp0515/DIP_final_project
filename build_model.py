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


def build_unet_reconstructor(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    
    # Decoder with Skip Connections
    u2 = layers.UpSampling2D((2, 2))(c3)
    u2 = layers.concatenate([u2, c2])
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    
    u1 = layers.UpSampling2D((2, 2))(c4)
    u1 = layers.concatenate([u1, c1])
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c5)
    
    outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(c5)
    
    return models.Model(inputs, outputs)