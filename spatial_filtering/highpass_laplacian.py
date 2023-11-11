import numpy as np
import matplotlib.pyplot as plt
import cv2


def zero_pad(image, padsize):
    new_x = 2 * padsize[0] + image.shape[0]
    new_y = 2 * padsize[1] + image.shape[1]
    
    new_image = np.zeros((new_x, new_y))
    new_image[padsize[0]: image.shape[0] + padsize[1], padsize[1]: image.shape[1] + padsize[1] ] = image
    return new_image



def hp_laplacian_filter(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])
    
    
    pad_x = kernel.shape[0] // 2
    pad_y = kernel.shape[1] // 2
    padded_image = zero_pad(image=image, padsize=(pad_x, pad_y))
    
    new_image = np.zeros_like(image, dtype=np.float32)
    
    for i in range(padded_image.shape[0] - 2 * pad_x):
        for j in range(padded_image.shape[1] - 2 * pad_y):
            x = kernel * padded_image[ i : i+kernel.shape[0], j : j+kernel.shape[1]]
            new_image[i, j] =  np.sum(x)
    new_image = new_image - np.min(new_image)
    new_image = new_image / np.max(new_image)
    new_image = new_image * 255
    return new_image.astype(np.uint8)



if __name__ == '__main__':
    image = cv2.imread('./images/blurry_moon.tif', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    
    plt.figure(figsize=(16, 16))
    plt.subplot(2,2,1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    
    filtered_image = hp_laplacian_filter(image)
    plt.subplot(2,2,2)
    plt.imshow(filtered_image, cmap='gray')
    plt.axis('off')
    plt.title(f'Laplasian Filtered Image')
    
    enhanced_image = filtered_image.astype(np.int32) + image.astype(np.int32)
    enhanced_image = enhanced_image / np.max(enhanced_image)
    print(enhanced_image.dtype)
    print(np.min(enhanced_image), np.max(enhanced_image))
    
    plt.subplot(2,2,3)
    plt.imshow(enhanced_image, cmap='gray')
    plt.axis('off')
    plt.title(f'Sharpened Image')

    
    plt.show()