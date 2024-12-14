import cv2
import numpy as np

def morph_enhance(img):
    kernel = np.ones((1,1), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    img_opening = cv2.morphologyEx(img_erosion, cv2.MORPH_OPEN, kernel)
    img_dilation = cv2.dilate(img_opening, kernel, iterations=1) 
    img_closing = cv2.morphologyEx(img_dilation, cv2.MORPH_CLOSE, kernel)

    return img_closing

