import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
from build_model import build_autoencoder, build_cnn_reconstructor, build_unet_reconstructor
from create_training_data import *

input_shape = (160, 160, 1)  # Grayscale image of size 160 * 160
model = build_unet_reconstructor(input_shape)
model.compile(optimizer='adam', loss='mse')
model.summary()

loaded_real  = np.load('./np_data/img_real.npy')

x = randomly_generate_testing_data(loaded_real/ 255, repeat=500)
y = np.repeat(loaded_real, 500, axis=0) / 255

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=5,
    validation_data=(X_test, y_test)
)

np.savez('loss.npz',
        training_loss = history.history['loss'],
        validate_loss = history.history['val_loss'])

model.save('model.h5')
