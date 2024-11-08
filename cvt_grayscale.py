import cv2
import numpy as np
import matplotlib.pyplot as plt


def cvt_gray(image: np.array):
    return np.mean(image, axis=2)

if __name__ == '__main__':
    image = cv2.imread('./images/cat.jpg')
    image = cv2.resize(image, (512, 512))
    gray_image = cvt_gray(image=image)
    
    
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original Image')
    
    plt.subplot(1,2,2)
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.title('Gray Image')
    
    plt.show()