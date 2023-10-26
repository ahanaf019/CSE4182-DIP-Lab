import cv2
import numpy as np
import matplotlib.pyplot as plt


def bit_image(image: np.array, bits: int):
    mask = pow(2, bits - 1)
    # print(mask)
    # mask = 0b11100000
    
    image = np.bitwise_and(image, mask) 
    return image

if __name__ == '__main__':
    image = cv2.imread('./images/cat.jpg')
    image = cv2.resize(image, (512, 512))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    plt.subplot(3,3,1)
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.title('Gray Image')
    

    for bit in range(1, 9):
        bit_img = bit_image(gray_image, bit)
        plt.subplot(3,3,bit+1)
        plt.imshow(bit_img, cmap='gray')
        plt.axis('off')
        plt.title(f'{bit} Bit image')
    plt.show()