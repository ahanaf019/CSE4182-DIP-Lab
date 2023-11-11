import matplotlib.pyplot as plt
import numpy as np
import cv2



def freq_transform(image):
    M = image.shape[0]
    N = image.shape[1]
    
    P = 2 * M
    Q = 2 * N
    
    image_p = np.zeros((P, Q), dtype=np.float64)
    image_p[0: M, 0: N] = image
    
    Mp = image_p.shape[0]
    Np = image_p.shape[1]
    x = np.linspace(0, Mp-1, Mp)
    x = np.array([x for i in range(Mp)])
    y = x.T

    pwr = np.power(-1, x+y)
    image_p = image_p * pwr   
    
    fft_img = np.fft.fft2(image_p)
    return fft_img


    
def normalize_freq_transform_image(image):
    tr_dft_img = log_transform(np.abs(image))

    tr_dft_img /= np.max(tr_dft_img)
    tr_dft_img *= 255
    tr_dft_img = tr_dft_img.astype(np.uint8)
    return tr_dft_img


def log_transform(image):
    return np.log(image + 1)


if __name__ == '__main__':
    image = cv2.imread('./images/vertical_rectangle.tif', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    
    image_f = freq_transform(image)
    
    