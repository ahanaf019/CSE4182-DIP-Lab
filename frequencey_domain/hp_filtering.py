import cv2
import numpy as np
import matplotlib.pyplot as plt


def freq_transformation(image):
    M = image.shape[0]
    N = image.shape[1]
    
    P = M * 2
    Q = N * 2
    
    image_p = np.zeros((P, Q))
    image_p[0 : M, 0 : N] = image
    
    x = np.linspace(0, P-1, P)
    x = np.array([x for i in range(Q)])
    
    y = np.linspace(0, Q-1, Q)
    y = np.array([y for i in range(P)])
    y = y.T
    
    image_p = image_p * np.power(-1, x+y)
    fft_image = np.fft.fft2(image_p)
    return fft_image


def inverse_transform(fft_image):
    P = fft_image.shape[0]
    Q = fft_image.shape[1]
    
    x = np.linspace(0, P-1, P)
    x = np.array([x for i in range(Q)])
    
    y = np.linspace(0, Q-1, Q)
    y = np.array([y for i in range(P)])
    y = y.T
    
    image = np.fft.ifft2(fft_image)
    image = np.real(image)
    image = image * np.power(-1, x+y)
    image = image[: int(P/2), : int(Q/2)]
    return image.astype(np.uint8)


def normalize_freq_transform(fft_image):
    fft_image = np.abs(fft_image)
    fft_image = log_transform(fft_image)
    
    fft_image = fft_image / np.max(fft_image)
    fft_image = fft_image * 255
    return fft_image.astype(np.uint8)

def get_D(fft_image):
    P = fft_image.shape[0]
    Q = fft_image.shape[1]
    
    x = np.linspace(0, P-1, P)
    x = np.array([x for i in range(Q)])
    
    y = np.linspace(0, Q-1, Q)
    y = np.array([y for i in range(P)])
    y = y.T
    
    x = np.power(x - P/2, 2)
    y = np.power(y - Q/2, 2)
    
    D = np.sqrt(x+y)
    return D
    

def IHP_filter(fft_image, cutoff):
    D = get_D(fft_image)
    
    H = D >= cutoff
    H = H.astype(np.uint8)
    return fft_image * H


def gaussian(fft_image, cutoff):
    D = get_D(fft_image)
    
    pwr = -1 * np.power(D, 2) / (2 * cutoff**2)
    H = 1 - np.power(np.exp(1), pwr)
    return fft_image * H


def butter_filter(fft_image, order, cutoff):
    D = get_D(fft_image)
    
    H = 1 / (1 + np.power(cutoff/ D, 2*order))
    return fft_image * H


def log_transform(image):
    return np.log(image + 1)


if __name__ == '__main__':
    image = cv2.imread('./images/blur_test.tif', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    
    
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    
    
    fft_image = freq_transformation(image)
    plt.figure()
    plt.imshow(normalize_freq_transform(fft_image), cmap='gray')
    plt.axis('off')
    plt.title('Frequency Domain')
    
    
    plt.figure()
    i = 1
    for cutoff in [10, 30, 60, 160, 260, 460]:
        filtered_image = IHP_filter(fft_image, cutoff)
        filtered_image = inverse_transform(filtered_image)
        
        plt.subplot(2, 3, i)
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'IHPF Cutoff: {cutoff}')
        plt.axis('off')
        i += 1
    
    plt.figure()
    i = 1
    for cutoff in [10, 30, 60, 160, 260, 460]:
        filtered_image = gaussian(fft_image, cutoff)
        filtered_image = inverse_transform(filtered_image)
        
        plt.subplot(2, 3, i)
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'Gaussian Cutoff: {cutoff}')
        plt.axis('off')
        i += 1
    
    plt.figure()
    i = 1
    for cutoff in [10, 30, 60, 160, 260, 460]:
        filtered_image = butter_filter(fft_image, 4, cutoff)
        filtered_image = inverse_transform(filtered_image)
        
        plt.subplot(2, 3, i)
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'Butter Cutoff: {cutoff}')
        plt.axis('off')
        i += 1
    
    plt.show()
    
    
    