import cv2
import matplotlib.pyplot as plt
import numpy as np


def to_bits(image, bits=8):

	image = image / 2 ** (8 - bits)
	# print(np.min(image), np.max(image))

	return image.astype(np.uint8)

if __name__ == '__main__':
	image = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)

	result7 = to_bits(image, bits=7)
	result6 = to_bits(image, bits=6)
	result4 = to_bits(image, bits=4)
	result2 = to_bits(image, bits=2)
	result1 = to_bits(image, bits=1)

	plt.subplot(3,2,1)
	plt.imshow(image, cmap='gray')
	plt.title('Original Image')
	plt.axis('off')
	plt.subplot(3,2,2)
	plt.imshow(result7, cmap='gray')
	plt.title('7bit Image')
	plt.axis('off')
	plt.subplot(3,2,3)
	plt.imshow(result6, cmap='gray')
	plt.title('6bit Image')
	plt.axis('off')
	plt.subplot(3,2,4)
	plt.imshow(result4, cmap='gray')
	plt.title('4bit Image')
	plt.axis('off')
	plt.subplot(3,2,5)
	plt.imshow(result2, cmap='gray')
	plt.title('2bit Image')
	plt.axis('off')
	plt.subplot(3,2,6)
	plt.imshow(result1, cmap='gray')
	plt.title('1bit Image')
	plt.axis('off')
	plt.show()

