import cv2
import matplotlib.pyplot as plt
import numpy as np

def to_bits(image, bits=8):

	image = image / 2 ** (8 - bits)
	# print(np.min(image), np.max(image))

	return image.astype(np.uint8)


if __name__ == "__main__":
    image = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
    
    mask = 0b11100000
    masked_image = np.bitwise_and(image, mask)
    sub_image = image - masked_image
    
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(masked_image, cmap='gray')
    plt.axis('off')
    plt.title('MSB Image')
    
    plt.subplot(1, 3, 3)
    plt.imshow(sub_image, cmap='gray')
    plt.axis('off')
    plt.title('Subtracted Image')
    plt.show()