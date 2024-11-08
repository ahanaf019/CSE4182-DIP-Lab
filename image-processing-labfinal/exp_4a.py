import numpy as np
import matplotlib.pyplot as plt
import cv2
import random 



def main():
    image = cv2.imread('./images/blur_test.tif', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    
    noisy_image = add_noise(image, 0.0)
    
    filtered_image = filter(noisy_image, cutoff=80)
    print(cv2.PSNR(image, noisy_image))
    print(cv2.PSNR(filtered_image, image))
    
def add_noise(image: np.ndarray, percentage=0.01):
    new_image = image.copy()
    
    noisy_count = int(image.shape[0] * image.shape[1] * percentage)
    vals = [0, 255]
    for _ in range(noisy_count):
        x = random.randint(0, image.shape[0] - 1)
        y = random.randint(0, image.shape[1] - 1)
        new_image[x, y] = np.random.choice(vals, 1)[0]
    return new_image


def filter(image, cutoff=100, type='gaussian'):
    P = image.shape[0]
    Q = image.shape[1]
    
    padded_image = np.zeros((P, Q))
    padded_image[0: image.shape[0], 0: image.shape[1]] = image
    
    x = np.array([[i for i in range(P)] for _ in range(P)])
    y = np.array([[j for _ in range(P)] for j in range(Q)])
    padded_image = np.power(-1, x + y) * padded_image
    
    dft_image = np.fft.fft2(padded_image)
    plt.figure()
    plt.imshow(log_transform(dft_image), cmap='gray')
    plt.title('Fourier Spectrum')
    
    D = np.sqrt(np.power(x - P/2, 2) + np.power(y - Q/2, 2))
    plt.figure()
    plt.imshow(D / np.max(D), cmap='gray')
    plt.title('D plot')
    filter_fn = gaussian_fn(D, cutoff)
    plt.figure()
    plt.imshow(filter_fn / np.max(filter_fn), cmap='gray')
    plt.title('Filter Function')
    
    filtered_image = dft_image * filter_fn
    sd_image = np.fft.ifft2(filtered_image) * np.power(-1, x + y)
    sd_image = sd_image[0: image.shape[0], 0: image.shape[1]]
    sd_image = np.abs(sd_image)
    sd_image = sd_image / np.max(sd_image)
    sd_image = sd_image * 255
    sd_image = sd_image.astype(np.uint8)
    plt.figure()
    plt.imshow(sd_image, cmap='gray')
    plt.title('Filtered Image')
    
    
    
    plt.show()
    return sd_image


def gaussian_fn(D, D0):
    x = np.power(D, 2) / (2 * np.power(D0, 2))
    return 1 - np.exp(-1 * x)


def log_transform(dft_image: np.ndarray):
    dft_image = np.abs(dft_image)
    dft_image = dft_image + 1e-7
    dft_image = np.log(dft_image)
    dft_image = dft_image / np.max(dft_image)
    dft_image = dft_image * 255
    return dft_image.astype(np.uint8)


if __name__ == '__main__':
    main()