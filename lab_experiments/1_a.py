import cv2
import matplotlib.pyplot as plt
import numpy as np


def downsample(image, factor=2):
	x_px = image.shape[0] // factor
	y_px = image.shape[1] // factor

	res_img = np.zeros(shape=(x_px, y_px))

	for i in range(x_px):
		for j in range(y_px):
			# res_img[i, j] = np.mean(image[i*factor: (i+1)*factor, j*factor: (j+1)*factor])
			res_img[i, j] = image[i*factor: (i+1)*factor, j*factor: (j+1)*factor][0][0]
	
	return res_img


if __name__ == '__main__':
	image = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)

	result1 = downsample(image, factor=2)
	result2 = downsample(image, factor=4)
	result3 = downsample(image, factor=64)

	plt.subplot(2,2,1)
	plt.imshow(image, cmap='gray')
	plt.title('Original Image')
	plt.axis('off')
	plt.subplot(2,2,2)
	plt.imshow(result1, cmap='gray')
	plt.title('Spatial Resolution 1/2')
	plt.axis('off')
	plt.subplot(2,2,3)
	plt.imshow(result2, cmap='gray')
	plt.title('Spatial Resolution 1/4')
	plt.axis('off')
	plt.subplot(2,2,4)
	plt.imshow(result3, cmap='gray')
	plt.title('Spatial Resolution 1/8')
	plt.axis('off')
	plt.show()

