import numpy as np
import matplotlib.pyplot as plt
import cv2


def zero_pad(image, padsize):
    new_x = 2 * padsize[0] + image.shape[0]
    new_y = 2 * padsize[1] + image.shape[1]
    
    new_image = np.zeros((new_x, new_y))
    new_image[padsize[0]: image.shape[0] + padsize[1], padsize[1]: image.shape[1] + padsize[1] ] = image
    return new_image


def arithmatic_mean(x):
    n = x.shape[0] * x.shape[1]
    return 1 / n * np.sum(x)

def geometric_mean(x):
    n = x.shape[0] * x.shape[1]
    x = x.astype(np.float32)
    x = np.log(x) / n
    x = np.sum(x)
    return np.power(np.exp(1), x) 
    
    

def harmonic_mean(x):
    return np.power(geometric_mean(x), 2) / arithmatic_mean(x)


def get_filter(type):
    filters = {
        'arithmatic': arithmatic_mean,
        'geometric': geometric_mean,
        'harmonic': harmonic_mean,
    }
    
    return filters[type]


def lp_mean_filter(image, kernel_size, type='arithmatic'):
    kernel = np.ones(kernel_size)
    
    filter_fn = get_filter(type)
    
    pad_x = kernel_size[0] // 2
    pad_y = kernel_size[1] // 2
    padded_image = zero_pad(image=image, padsize=(pad_x, pad_y))
    
    new_image = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(padded_image.shape[0] - 2 * pad_x):
        for j in range(padded_image.shape[1] - 2 * pad_y):
            x = kernel * padded_image[ i : i+kernel_size[0], j : j+kernel_size[1]]
            new_image[i, j] =  filter_fn(x)
    return new_image



if __name__ == '__main__':
    image = cv2.imread('./images/circuit-board-salt-prob-pt1.tif', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    
    plt.figure(figsize=(16, 16))
    plt.subplot(2,2,1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    
    i = 2
    k = 3
    for type in ['arithmatic', 'geometric', 'harmonic']:
        filtered_image = lp_mean_filter(image, (k,k), type=type)
        plt.subplot(2,2,i)
        plt.imshow(filtered_image, cmap='gray')
        plt.axis('off')
        plt.title(f'({k}x{k}) {type} filter')
        i += 1
    
    plt.show()