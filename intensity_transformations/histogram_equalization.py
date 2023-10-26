import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_histogram(image: np.array):
    x = np.array([ i for i in range(256) ])
    y = np.array([ 0 for i in range(256) ], dtype=np.int32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            y[image[i, j]] += 1

    return x, y


def equalize_hist(image: np.array):
    L = 256
    x, y = get_histogram(image)
    mn = image.shape[0] * image.shape[1]
    
    pr = y / mn
    s = []
    sk = 0
    for i in range(L):
        sk += pr[i]
        s.append(np.round(sk * (L - 1)))
    
    s = np.array(s)
    
    eq_image = apply_hist(image, s)
    return eq_image


def apply_hist(image, hist_f):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            intensity = image[i, j]
            image[i, j] = hist_f[intensity]
    return image
    

if __name__ == '__main__':
    image = cv2.imread('./images/stone.tif')
    gray_image = cv2.resize(image, (512, 512))
    
    # gray_image = cv2.cvtColor(image)
    
    
    x, y = get_histogram(gray_image)
    plt.subplot(2, 1, 1)
    plt.bar(x, y)
    plt.xlim([0, 255])
    plt.axis('on')
    plt.title(f'Histogram')
    
    eq_image = equalize_hist(gray_image.copy())
    x, y = get_histogram(eq_image)
    plt.subplot(2, 1, 2)
    plt.bar(x, y)
    plt.xlim([0, 255])
    plt.axis('on')
    plt.title(f'Equalized Histogram')
    
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.title(f'Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(eq_image, cmap='gray')
    plt.axis('off')
    plt.title(f'Histogram Equalized Image')
    plt.show()
    