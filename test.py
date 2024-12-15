import numpy as np
import matplotlib.pyplot as plt


loaded_real = np.load('../dataset/np_data/img_real.npy')
print(loaded_real.shape)

loaded_train = np.load('../dataset/np_data/img_train.npy')
print(loaded_train.shape)

loaded_label_real = np.load('../dataset/np_data/label_real.npy')
print(loaded_label_real.shape)

loaded_label_train = np.load('../dataset/np_data/label_train.npy')
print(loaded_label_train.shape)




fig, ax = plt.subplots(1, 10, figsize=(15, 3))

for i in range(10):
    ax[i].imshow(loaded_real[i, :, :, 0], cmap='gray')
    ax[i].set_xticks([])
    ax[i].set_yticks([])



plt.show()