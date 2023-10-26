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


def single_threshold_segment(image, threshold=127):
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if image[i, j] > threshold:
				image[i, j] = 1
			else:
				image[i, j] = 0
	return image

if __name__ == '__main__':
	img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)

	hist = histogram(img)
	segmented_image = single_threshold_segment(img, threshold=150)
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
	plt.imshow(segmented_image, cmap='gray')
	plt.title('Segmented Image')
	plt.axis('off')
	
	plt.show()

