import cv2
import matplotlib.pyplot as plt
import numpy as np


def histogram(image):
	y = [ 0 for i in range(256) ]
	x = [ i for i in range(256) ]
	
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			y[image[i,j]] += 1

	return x, y


def enhance_brightness(image, inc, rng=(0, 255)):
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if image[i, j] >= rng[0] and image[i, j] <= rng[1]:
				image[i, j] += inc
	image[ image > 255] = 255
	image[ image < 0] = 0
	return image

if __name__ == '__main__':
	img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)

	hist = histogram(img)
	enhanced_image = enhance_brightness(img, 50, (150, 205))
	img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)

	plt.subplot(1,3,1)
	plt.imshow(img, cmap='gray')
	plt.title('Original Image')
	plt.axis('off')
	plt.subplot(1,3,2)
	plt.bar(hist[0], hist[1])
	plt.title('Histogram of the Image')
	plt.axis('on')
	plt.subplot(1,3,3)
	plt.imshow(enhanced_image, cmap='gray')
	plt.title('Enhanced Image')
	plt.axis('off')
	
	plt.show()

