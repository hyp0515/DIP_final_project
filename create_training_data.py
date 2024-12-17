import numpy as np
import cv2
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
    img = np.clip(img + np.random.normal(0, 0.1, img.shape), 0., 1.)
    return img

def median(img):
    img = median_filter(img, size=3)
    return np.clip(img, 0, 1)

def bilateral_detail(img):  #保留細節, 圖像模糊
    img = img.astype(np.float32)
    img = cv2.bilateralFilter(img,9,10,75)  #小鄰域 = 9, 高顏色敏感度 = 10, 中等空間範圍 = 75    
    return np.clip(img, 0, 1)

def bilateral_blur(img):  #平滑程度明顯增強，更多細節被模糊化，但邊緣仍保留。
    img = img.astype(np.float32)
    img = cv2.bilateralFilter(img,20,10,150)      
    return np.clip(img, 0, 1)

def bilateral_edge(img):  #邊緣清晰，但色彩分佈可能變得不自然。
    img = img.astype(np.float32)
    img = cv2.bilateralFilter(img,9,100,10)      
    return np.clip(img, 0, 1)

def bilateral_smooth(img):  #濾波範圍更大，細節被平滑化，但邊緣仍保留。
    img = img.astype(np.float32)
    img = cv2.bilateralFilter(img,9,100,50)      
    return np.clip(img, 0, 1)

def randomly_generate_testing_data(real_dataset, repeat=100):
    
    n_samples, height, width, channels = real_dataset.shape
    
    testing_dataset = np.empty((n_samples * repeat, height, width, channels))

    augmentations = [
        blur,
        blur,
        sharpen,
        sharpen,
        noise,
        noise,
        noise,
        noise,
        noise,
        # bilateral_smooth,
        lambda img: img,  # Original
        lambda img: img,  # Original
    ]
            
    for i in range(n_samples):
        for j in range(repeat):
            method_idx = np.random.randint(0, len(augmentations))
            augmented_img = augmentations[method_idx](real_dataset[i, :, :, 0])
            testing_dataset[i * repeat + j, :, :, 0] = augmented_img
            
    return testing_dataset
