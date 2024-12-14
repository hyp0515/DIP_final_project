import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from autoencoder import build_autoencoder
from initialize_image import preprocess_fingerprint_dataset
from enhance_fingerprint import enhance_fingerprints

# Initialize the model
input_shape = (128, 128, 1)  # Grayscale image of size 128x128
autoencoder = build_autoencoder(input_shape)

# def ssim_loss(y_true, y_pred):
#     return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

fingerprint_path = "./DB4_B/"
preprocess_fingerprint_dataset(input_dir=fingerprint_path,
                              output_fname='preprocessed_fingerprints',
                              output_size=(128,128))

dataset = np.load('preprocessed_fingerprints.npy')

n_fingers, n_impressions = dataset.shape[:2]

# Training set: first 6 impressions
X_train = dataset[:, :6, :, :, :]  # Shape: (n_fingers, 6, 128, 128, 1)

# Testing set: last 2 impressions
X_test = dataset[:, 6:, :, :, :]  # Shape: (n_fingers, 2, 128, 128, 1)

# Reshape to collapse the first two dimensions for input into the autoencoder
X_train = X_train.reshape(-1, 128, 128, 1)  # Shape: (n_fingers * 6, 128, 128, 1)
X_test = X_test.reshape(-1, 128, 128, 1)    # Shape: (n_fingers * 2, 128, 128, 1)

Y_train = np.clip(enhance_fingerprints(X_train), 0., 1.)
Y_test = np.clip(enhance_fingerprints(X_test), 0., 1.)

X_train = np.clip(X_train + np.random.normal(0, 0.05, X_train.shape), 0., 1.)
X_test = np.clip(X_test + np.random.normal(0, 0.05, X_test.shape), 0., 1.)

history = autoencoder.fit(
    X_train, Y_train,
    epochs=10,
    batch_size=1,
    validation_data=(X_test, Y_test)
)
autoencoder.save('model.h5')

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()


load_trained = load_model('model.h5')
reconstructed_images = load_trained.predict(X_test)

# print(reconstructed_images.shape)
n = 5  # Number of samples to display
fig, ax = plt.subplots(3, n, figsize=(12, 8), sharex=True, sharey=True)
fig.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1, wspace=0.0, hspace=-0.05)
for i in range(n):
    ax[0, i].imshow(X_test[i, :, :, 0].reshape(128, 128), cmap='gray')
    ax[0, 0].set_ylabel("Noised")
    ax[0, i].set_yticks([])
    ax[0, i].set_xticks([])
    
    ax[1, i].imshow(reconstructed_images[i, :, :, 0].reshape(128, 128), cmap='gray')
    ax[1, 0].set_ylabel("Reconstructed")
    ax[1, i].set_yticks([])
    ax[1, i].set_xticks([])
    
    ax[2, i].imshow(Y_test[i, :, :, 0].reshape(128, 128), cmap='gray')
    ax[2, 0].set_ylabel("Truth")
    ax[2, i].set_yticks([])
    ax[2, i].set_xticks([])
plt.savefig('compare.png')
# plt.show()

