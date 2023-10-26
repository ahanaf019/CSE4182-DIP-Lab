import cv2
import matplotlib.pyplot as plt
import numpy as np


def power_law_transform(image, gamma):
    image = image.astype(np.float32)
    image = image / 255
    image = 255 * image ** gamma
    return image.astype(np.uint8)


def log_transform(image):
    image = image.astype(np.float32)
    c = 255 / np.log(1 + np.max(image))
    image = c * np.log(1 + image)
    return image.astype(np.uint8)


if __name__ == "__main__":
    image = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
    
    plt.subplot(1,5, 1)
    plt.imshow(image, cmap="gray")
    plt.title('Original Image')
    plt.axis('off')
    
    
    i = 2
    for gamma in [0.1, 0.5, 1.2, 2.2]:
        # Apply gamma correction.
        gamma_corrected = log_transform(image)
    
        plt.subplot(1,5, i)
        plt.imshow(gamma_corrected, cmap='gray')
        plt.axis('off')
        plt.title(f'Gamma {gamma}')
        i += 1
    plt.show()