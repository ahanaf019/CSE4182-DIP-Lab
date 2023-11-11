import matplotlib.pyplot as plt
import numpy as np
import cv2


def dft_step(image, u, v, x, y):
    M = image.shape[0]
    N = image.shape[1]

    ex = np.power(np.exp(1), -1j * 2 * np.pi * (u * x + v * y))
    res = np.sum(image * ex)
    
    return res


def dft(image):
    M = image.shape[0]
    N = image.shape[1]
    
    image = image.astype(np.float32)
    dft_img = np.zeros_like(image, dtype=np.complex64)
    
    x = np.linspace(0, M-1, M)
    x = np.array([x for i in range(M)]) / M
    y = x.T
    
    for u in range(M):
        print(u)
        for v in range(N):
            dft_img[u, v] = dft_step(image, u, v, x, y)

    return dft_img


def log_transform(image):
    return np.log(image + 1)


if __name__ == '__main__':
    image = cv2.imread('./images/applo17_boulder_noisy.tif', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    
    # dft_img = dft(image)
    # print(np.min(dft_img), np.max(dft_img))
    
    x = np.linspace(0, image.shape[0]-1, image.shape[0])
    x = np.array([x for i in range(image.shape[0])]) 
    y = x.T
    
    fft_img = np.fft.fft2(image)
    print(fft_img.shape)
    fft_img = fft_img * np.power(-1, x+y)
    print(np.min(fft_img), np.max(fft_img))
    
    plt.subplot(1,3,1)
    plt.imshow(image, cmap='gray')
    
    # tr_dft_img = log_transform(np.abs(dft_img))
    # tr_dft_img /= np.max(tr_dft_img)
    # tr_dft_img *= 255
    # tr_dft_img = tr_dft_img.astype(np.uint8)
    
    # print(np.min(tr_dft_img), np.max(tr_dft_img))
    # plt.subplot(1,3,2)
    # plt.imshow(tr_dft_img, cmap='gray')
    
    
    tr_dft_img = log_transform(np.abs(fft_img))
    print(np.min(tr_dft_img), np.max(tr_dft_img))
    
    print(np.min(tr_dft_img), np.max(tr_dft_img))
    
    tr_dft_img /= np.max(tr_dft_img)
    tr_dft_img *= 255
    tr_dft_img = tr_dft_img.astype(np.uint8)
    print(np.min(tr_dft_img), np.max(tr_dft_img))
    plt.subplot(1,3,3)
    plt.imshow(tr_dft_img, cmap='gray')
    
    plt.show()

    