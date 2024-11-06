import cv2
import numpy as np
import matplotlib.pyplot as plt


def freq_transform(image):
    M = image.shape[0]
    N = image.shape[1]
    
    P = M * 2
    Q = N * 2
    
    image_p = np.zeros((P, Q))
    image_p[0 : M, 0 : N] = image
    
    x = np.linspace(0, P-1, P)
    x = np.array([x for i in range(P)])
    
    y = np.linspace(0, Q-1, Q)
    y = np.array([y for i in range(Q)])
    y = y.T
    
    image_p = image_p * np.power(-1, x+y)
    
    fft_image = np.fft.fft2(image_p)
    return fft_image


def inverse_transform(freq_image):
    P = freq_image.shape[0]
    Q = freq_image.shape[1]
    
    x = np.linspace(0, P-1, P)
    x = np.array([x for i in range(P)])
    
    y = np.linspace(0, Q-1, Q)
    y = np.array([y for i in range(Q)])
    y = y.T
    
    image = np.fft.ifft2(freq_image)
    image = image * np.power(-1, x+y)
    image = image[ : int(P/2), : int(Q/2)]
    return np.real(image)


    
def normalize_freq_transform(image):
    image = log_transform(np.abs(image))
    image = image / np.max(image)
    image = image * 255
    return image.astype(np.uint8)


def log_transform(image):
    return np.log(image + 1)


def get_D(freq_image):
    P = freq_image.shape[0]
    Q = freq_image.shape[1]
    
    u = np.linspace(0, P-1, P)
    u = np.array([ u for i in range(Q) ])
    u = (u - (P / 2))**2
    
    v = np.linspace(0, Q-1, Q)
    v = np.array([ v for i in range(P) ])
    v = (v - (Q / 2))**2
    v = v.T
    
    D = np.sqrt(u+v)
    return D


def butter_filter(freq_image, order, cutoff):
    D = get_D(freq_image)
    
    H = 1 / (1 + D/cutoff)**order
    return freq_image * H
    

def gaussian_filter(freq_image, cutoff):
    D = get_D(freq_image)
    pwr = -1 * np.power(D, 2) / (2 * cutoff)**2
    
    H = np.power(np.exp(1), pwr)
    return freq_image * H


def ILP_filter(freq_image, cutoff=10):
    D = get_D(freq_image)
    
    H = D < cutoff
    H = H.astype(np.uint8)
    # print(H)
    
    return freq_image * H




if __name__ == '__main__':
    image = cv2.imread('./images/blur_test.tif', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    
    
    fft_image = freq_transform(image)
    print(fft_image.shape)
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    plt.figure()
    plt.imshow(normalize_freq_transform(fft_image), cmap='gray')
    plt.title('Fourier Spectrum')
    
    plt.figure()
    i = 1
    for cutoff in [10, 30, 60, 130, 260, 460]:
        filtered_image = ILP_filter(fft_image, cutoff)
        filtered_image = inverse_transform(filtered_image)
        
        plt.subplot(2, 3, i)
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'ILPF Cutoff: {cutoff}')
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
        
    plt.figure()
    i = 1
    for cutoff in [10, 30, 60, 160, 260, 460]:
        filtered_image = gaussian_filter(fft_image, cutoff)
        filtered_image = inverse_transform(filtered_image)
        
        plt.subplot(2, 3, i)
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'Gaussian Cutoff: {cutoff}')
        plt.axis('off')
        i += 1
    
    plt.show()