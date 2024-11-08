import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

def apply_salt_pepper_noise(image, noise_percentage):
    noisy_image = np.copy(image)

    # Calculate the number of pixels to be affected
    num_pixels = int(noise_percentage * image.size)
    print(num_pixels)
    print(noisy_image)
    # Create random indices for salt and pepper noise
    vals = [0, 255]
    for i in range(num_pixels):
        x = random.randint(0, noisy_image.shape[0] - 1)
        y = random.randint(0, noisy_image.shape[1] - 1)
        noisy_image[x, y] = np.random.choice(vals, 1)[0]
    return noisy_image

# Example usage:
# Load an image using OpenCV
input_image = cv2.imread('./images/blur_test.tif', cv2.IMREAD_GRAYSCALE)

# Convert the image to grayscale (optional)
gray_image = input_image
# gray_image = gray_image.astype(np.float32) / 255.0
# gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
# print(gray_image)
# Apply salt and pepper noise with a noise percentage of 0.01 (1%)
noisy_image = apply_salt_pepper_noise(gray_image, 0.01)

# Display the original and noisy images
plt.figure()
plt.subplot(1,2,1)
plt.imshow(gray_image, cmap='gray')
# plt.imshow(input_image, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(noisy_image, cmap='gray')
plt.show()

