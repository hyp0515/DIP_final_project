import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

def blur(img):
    kernel = 1/9 * np.ones((3, 3))
    img = cv2.filter2D(img, -1, kernel)
    return np.clip(img, 0, 1)

def sharpen(img):
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    img = cv2.filter2D(img, -1, kernel)
    return np.clip(img, 0, 1)

def noise(img):
    img = np.clip(img + np.random.normal(0, 0.05, img.shape), 0., 1.)
    return img

def median(img):
    img = median_filter(img, size=3)
    return np.clip(img, 0, 1)

def randomly_generate_testing_data(real_dataset, repeat=100):
    
    n_samples, height, width, channels = real_dataset.shape
    
    testing_dataset = np.empty((n_samples * repeat, height, width, channels))

    augmentations = [
        # blur,
        # sharpen,
        # noise,
        # median,
        # lambda img: blur(sharpen(img)),  # Blur + Sharpen
        # lambda img: blur(noise(img)),    # Blur + Noise
        # lambda img: sharpen(noise(img)), # Sharpen + Noise
        # lambda img: blur(sharpen(noise(img))),  # Blur + Sharpen + Noise
        # lambda img: median(noise(img)),
        lambda img: img,  # Original
        lambda img: img,  # Original
        lambda img: img,  # Original
        lambda img: img,  # Original
        lambda img: img,  # Original
        lambda img: img,  # Original
        lambda img: img,  # Original
        lambda img: img,  # Original
    ]
            
    for i in range(n_samples):
        for j in range(repeat):
            method_idx = np.random.randint(0, len(augmentations))
            augmented_img = augmentations[method_idx](real_dataset[i, :, :, 0])
            testing_dataset[i * repeat + j, :, :, 0] = augmented_img
            
    return testing_dataset
