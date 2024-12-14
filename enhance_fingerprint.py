import numpy as np
from enhance.enhance_package import image_enhance
# def enhance_fingerprints(dataset):
#     enhanced_dataset = np.empty(dataset.shape)
#     for i in range(dataset.shape[0]):
#         enhanced_dataset[i, :, :, 0] = image_enhance.image_enhance(dataset[i, :, :, 0])
#     return enhanced_dataset

from enhance.morph_enhance import morph_enhance
def enhance_fingerprints(dataset):
    enhanced_dataset = np.empty(dataset.shape)
    for i in range(dataset.shape[0]):
        enhanced_dataset[i, :, :, 0] = morph_enhance(dataset[i, :, :, 0])
    return enhanced_dataset

