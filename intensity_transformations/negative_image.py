import cv2
import numpy as np
import matplotlib.pyplot as plt


def cvt_negative(image: np.array):
    return 255 - image

if __name__ == '__main__':
    image = cv2.imread('./images/cat.jpg')
    image = cv2.resize(image, (512, 512))
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    neg_image = cvt_negative(image=gray_image)
    
    
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original Image')
    
    plt.subplot(1,3,2)
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.title('Gray Image')
    
    plt.subplot(1,3,3)
    plt.imshow(neg_image, cmap='gray')
    plt.axis('off')
    plt.title('Negative Image')
    
    plt.show()