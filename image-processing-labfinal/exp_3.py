import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import math
import sklearn


def geometric_mean(subsec: np.ndarray):
    n  =subsec.shape[0] * subsec.shape[1]
    x = subsec.astype(np.float64)
    x = np.prod(x)
    x = np.power(x, 1/n)
    return x.astype(np.uint8)

def harmonic_mean(subsec: np.ndarray):
    n  =subsec.shape[0] * subsec.shape[1]
    x = subsec.astype(np.float64)
    x = 1 / (subsec + 1e-6)
    x = np.sum(x) 
    x = n / x
    return x.astype(np.uint8)

def filter(image: np.ndarray, kernel_size=(3,3), type='average'):
    if type == 'average':
        fn = np.mean
    if type == 'median':
        fn = np.median
    if type == 'geometric':
        fn = geometric_mean
    if type == 'harmonic':
        fn = harmonic_mean
    kernel = np.ones(kernel_size)
    
    padx = kernel_size[0] // 2
    pady = kernel_size[1] // 2
    padded_image = np.zeros((image.shape[0] + 2 * padx, image.shape[1] + 2 * pady), dtype=np.float32)
    
    padded_image[padx:image.shape[0] + padx, padx:image.shape[1] + pady] = image
    new_image = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            subsec = padded_image[i : i + kernel_size[0], j : j + kernel_size[0]]
            new_image[i, j] = fn(subsec * kernel)
    return new_image.astype(np.uint8)
    


def add_noise(image: np.ndarray, percentage=0.01):
    image = image.copy()
    noisy_pixels = int(image.shape[0] * image.shape[1] * percentage)
    vals = [0, 255]
    
    for i in range(noisy_pixels):
        x = random.randint(0, image.shape[0] - 1)
        y = random.randint(0, image.shape[1] - 1)
        
        image[x, y] = np.random.choice(vals, 1)[0]
    return image


def psnr(image : np.ndarray, noisy_image: np.ndarray):
    image = image.astype(np.float32)
    noisy_image = noisy_image.astype(np.float32)
    return 10 * math.log10(256**2 / np.mean(np.power(image - noisy_image, 2)))


def main():
    image = cv2.imread('./images/cat.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    
    # noisy_image = add_noise(image, 0.01)
    # filtered_image = filter(noisy_image, kernel_size=(5,5), type='average')
    # plt.figure()
    # plt.subplot(1,3,2)
    # plt.imshow(noisy_image, cmap='gray')
    # plt.subplot(1,3,1)
    # plt.imshow(image, cmap='gray')
    # # plt.show()

    # print(cv2.PSNR(image, noisy_image))
    # print(cv2.PSNR(filtered_image, noisy_image))
    # plt.subplot(1,3,3)
    # plt.imshow(filtered_image, cmap='gray')
    
    # filtered_image = filter(noisy_image, kernel_size=(5,5), type='median')
    # plt.figure()
    # plt.subplot(1,3,2)
    # plt.imshow(noisy_image, cmap='gray')
    # plt.subplot(1,3,1)
    # plt.imshow(image, cmap='gray')
    # print(cv2.PSNR(image, noisy_image))
    # print(cv2.PSNR(filtered_image, noisy_image))
    # plt.subplot(1,3,3)
    # plt.imshow(filtered_image, cmap='gray')
    # plt.show()
    
    noisy_image = add_noise(image, 0.01)
    
    plt.subplot(1,5,1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1,5,2)
    plt.imshow(noisy_image, cmap='gray')
    print(cv2.PSNR(image, noisy_image))
    # print(psnr(image.copy(), image.copy()))
    idx = 3
    for i in [3, 5, 7]:
        filtered_image = filter(noisy_image, kernel_size=(i,i), type='average')
        print(cv2.PSNR(filtered_image, image))
        plt.subplot(1,5,idx)
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'({i},{i})')
        idx += 1
    plt.show()
    
if __name__ == '__main__':
    main()