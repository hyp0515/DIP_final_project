import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

from build_model import build_autoencoder, build_cnn_reconstructor, build_unet_reconstructor
from initialize_image import preprocess_fingerprint_dataset
from enhance_fingerprint import enhance_fingerprints
from create_training_data import randomly_generate_testing_data, sharpen

input_shape = (160, 160, 1)  # Grayscale image of size 160 * 160
model = build_unet_reconstructor(input_shape)
# model = build_autoencoder(input_shape)
model.compile(optimizer='adam', loss='mse')
model.summary()


loaded_real  = np.load('./np_data/img_real.npy')

x = randomly_generate_testing_data(loaded_real/ 255, repeat=100)
y = np.repeat(loaded_real, 100, axis=0) / 255

print(x.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# history = model.fit(
#     X_train, y_train,
#     epochs=5,
#     batch_size=2,
#     validation_data=(X_test, y_test)
# )
# model.save('model.h5')

# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.legend()
# plt.savefig('train_loss.png')
# plt.close('all')


# 加載模型
load_trained = load_model('model.h5')

# 單一圖片路徑
fingerprint_path = "./DB4_B/102_1.tif"

# 圖片大小設定
output_size = (160, 160)

# 加載與處理單一圖片
def process_single_image(image_path, output_size):
    img = Image.open(image_path).convert('L')  # 將圖片轉為灰階
    img_resized = img.resize(output_size)
    img_array = np.array(img_resized) / 255 # 歸一化到 [0, 1]
    return img_array

# 處理圖片
processed_image = process_single_image(fingerprint_path, output_size=(200, 200))
processed_image = processed_image[20:-20, 20:-20]
processed_image = np.expand_dims(processed_image, axis=(0, -1))  # Shape: (1, 160, 160, 1)
# processed_image = enhance_fingerprints(processed_image)
# processed_image[0, :, :, 0] = sharpen(processed_image[0, :, :, 0])
# 預測
noised_img = np.clip(processed_image + np.random.normal(0, 0.05, processed_image.shape), 0., 1.)
# noised_img = processed_image
reconstructed_image = load_trained.predict(noised_img)
reconstructed_image[0, :, :, 0] = sharpen(reconstructed_image[0, :, :, 0])
# 將圖片可視化
fig, ax = plt.subplots(3, 1, figsize=(5, 10), sharex=True, sharey=True)
ax[0].imshow(noised_img[0, :, :, 0], cmap='gray')
ax[0].set_title("Input (Noised)")
ax[0].axis('off')

ax[1].imshow(reconstructed_image[0, :, :, 0], cmap='gray')
ax[1].set_title("Reconstructed")
ax[1].axis('off')

# 如果有 ground truth 可用，則繪製真實圖片
# 此處假設 y_test[0] 是 ground truth
ax[2].imshow(processed_image[0, :, :, 0], cmap='gray')
ax[2].set_title("Ground Truth")
ax[2].axis('off')

plt.tight_layout()
# plt.show()
plt.savefig('compare_single.png')
plt.close('all')