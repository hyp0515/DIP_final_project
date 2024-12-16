import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


from build_model import build_autoencoder, build_cnn_reconstructor
from initialize_image import preprocess_fingerprint_dataset
from enhance_fingerprint import enhance_fingerprints

# Initialize the model
input_shape = (160, 160, 1)  # Grayscale image of size 128x128
model = build_autoencoder(input_shape)
model.compile(optimizer='adam', loss='mse')
model.summary()

loaded_real  = np.load('./np_data/img_real.npy')
loaded_train = np.load('./np_data/img_train.npy')

print(loaded_real.shape)
print(loaded_train.shape)


x = np.clip(loaded_train / 255 + np.random.normal(0, 0.1, loaded_train.shape), 0., 1.)
# x = loaded_train / 255
y = loaded_train / 255

print(x.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=5,
    validation_data=(X_test, y_test)
)
model.save('model.h5')

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.savefig('train_loss.png')
plt.close('all')


load_trained = load_model('model.h5')
reconstructed_images = load_trained.predict(X_test)

# print(reconstructed_images.shape)
n = 5  # Number of samples to display
fig, ax = plt.subplots(3, n, figsize=(12, 8), sharex=True, sharey=True)
fig.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1, wspace=0.0, hspace=-0.05)
for i in range(n):
    ax[0, i].imshow(X_test[i, :, :, 0].reshape(160, 160), cmap='gray')
    ax[0, 0].set_ylabel("Noised")
    ax[0, i].set_yticks([])
    ax[0, i].set_xticks([])
    
    ax[1, i].imshow(reconstructed_images[i, :, :, 0].reshape(160, 160), cmap='gray')
    ax[1, 0].set_ylabel("Reconstructed")
    ax[1, i].set_yticks([])
    ax[1, i].set_xticks([])
    
    ax[2, i].imshow(y_test[i, :, :, 0].reshape(160, 160), cmap='gray')
    ax[2, 0].set_ylabel("Truth")
    ax[2, i].set_yticks([])
    ax[2, i].set_xticks([])
plt.savefig('compare.png')
plt.close('all')
# plt.show()

