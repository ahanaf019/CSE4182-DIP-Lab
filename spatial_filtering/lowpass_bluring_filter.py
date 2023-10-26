import numpy as np
import matplotlib.pyplot as plt
import cv2


def zero_pad(image, padsize):
    new_x = 2 * padsize[0] + image.shape[0]
    new_y = 2 * padsize[1] + image.shape[1]
    
    new_image = np.zeros((new_x, new_y))
    new_image[padsize[0]: image.shape[0] + padsize[1], padsize[1]: image.shape[1] + padsize[1] ] = image
    return new_image


def lp_average_filter(image, kernel_size):
    kernel = np.ones(kernel_size)
    
    pad_x = kernel_size[0] // 2
    pad_y = kernel_size[1] // 2
    padded_image = zero_pad(image=image, padsize=(pad_x, pad_y))
    
    new_image = np.zeros_like(image)
    
    for i in range(padded_image.shape[0] - 2 * pad_x):
        for j in range(padded_image.shape[1] - 2 * pad_y):
            x = kernel * padded_image[ i : i+kernel_size[0], j : j+kernel_size[1]]
            new_image[i, j] =  1 / np.sum(kernel) * np.sum(x)
    return new_image



if __name__ == '__main__':
    image = cv2.imread('./images/blur_test.tif', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    
    plt.figure(figsize=(16, 16))
    plt.subplot(2,3,1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    
    i = 2
    for k in [3, 5, 9, 15, 35]:
        filtered_image = lp_average_filter(image, (k,k))
        plt.subplot(2,3,i)
        plt.imshow(filtered_image, cmap='gray')
        plt.axis('off')
        plt.title(f'({k}x{k}) mask')
        i += 1
    
    plt.show()