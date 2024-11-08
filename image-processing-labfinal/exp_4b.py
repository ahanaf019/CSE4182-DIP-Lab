import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    image = cv2.imread('./images/blur_test.tif', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    ideal_lp(image, cutoff=460)


def pad_image(image: np.ndarray, p , q) -> np.ndarray:
    padded_image = np.zeros((p,q))
    padded_image[0: image.shape[0], 0: image.shape[1]] = image
    return padded_image

def log_transform(image: np.ndarray):
    # print(image.min(), image.max())
    x = np.log(image + 1)
    x = x / x.max()
    x = x * 255
    return x.astype(np.uint8)


def ideal_lp(image: np.ndarray, cutoff=30):
    P = 2 * image.shape[0]
    Q = 2 * image.shape[1]
    padded_image = pad_image(image, P, Q)
    # print(padded_image)
    x = np.array([[i for i in range(P)] for _ in range(P)])
    y = np.array([[j for _ in range(Q)] for j in range(Q)])
    # print(x + y)
    padded_image = np.power(-1, x + y) * padded_image
    # print(padded_image)
    dft_image = np.fft.fft2(padded_image)
    # plt.imshow(log_transform(np.abs(dft_image)), cmap='gray')
    
    D = np.sqrt(np.power(x - P/2, 2) + np.power(y - Q/2, 2))
    # print(D)
    # plt.imshow(D / np.max(D), cmap='gray')
    
    H = D < cutoff
    H = H.astype(np.uint8)
    # H = H.astype(np.float32)
    # H = H * D
    # plt.imshow(H / np.max(H), cmap='gray')
    
    filtered_image = H * dft_image
    # plt.imshow(log_transform(np.abs(filtered_image)), cmap='gray')
    
    sd_image = np.fft.ifft2(filtered_image) * np.power(-1, x + y)
    sd_image = np.abs(sd_image)
    sd_image = sd_image[0: image.shape[0], 0: image.shape[1]]
    print(sd_image.min(), sd_image.max())
    plt.imshow(sd_image / np.max(sd_image), cmap='gray')
    
    plt.show()
    
    

if __name__ == '__main__':
    main()

