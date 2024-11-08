import numpy as np
import matplotlib.pyplot as plt
import cv2


def decrease_spatial_resoution(image, times=2):
    new_image = np.zeros((image.shape[0] // times, image.shape[1] // times))
    
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            new_image[i, j] = image[i*times, j*times]
    return new_image

def decrease_intensity_resoution(image, bits=8):
    new_image = image / np.power(2, 8 - bits)
    return new_image.astype(np.uint8)


def histogram(image: np.ndarray):
    histogram = np.zeros((256))
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            histogram[image[i, j]] += 1
    return histogram

def segment(image: np.ndarray, threshold=128):
    new_image = image >= threshold
    return new_image.astype(np.uint8)


def main():
    image = cv2.imread('./images/cat.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    
    # plt.figure()
    # new_image = decrease_spatial_resoution(image, times=2)
    # plt.subplot(2,2,1)
    # plt.imshow(image, cmap='gray')
    
    # plt.subplot(2,2,2)
    # plt.imshow(new_image, cmap='gray')
    
    # new_image = decrease_spatial_resoution(image, times=4)
    # plt.subplot(2,2,3)
    # plt.imshow(new_image, cmap='gray')
    
    # new_image = decrease_spatial_resoution(image, times=8)
    # plt.subplot(2,2,4)
    # plt.imshow(new_image, cmap='gray')
    # plt.show()


    # plt.figure()
    # new_image = decrease_intensity_resoution(image, bits=3)
    # plt.subplot(2,2,1)
    # plt.imshow(image, cmap='gray')
    
    # plt.subplot(2,2,2)
    # plt.imshow(new_image, cmap='gray')
    
    # new_image = decrease_intensity_resoution(image, bits=2)
    # plt.subplot(2,2,3)
    # plt.imshow(new_image, cmap='gray')
    
    # new_image = decrease_intensity_resoution(image, bits=1)
    # plt.subplot(2,2,4)
    # plt.imshow(new_image, cmap='gray')
    # plt.show()
    
    hist = histogram(image)
    seg = segment(image, threshold=150)
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1,2,2)
    plt.plot(hist)
    plt.subplot(2,2,3)
    plt.imshow(seg, cmap='gray')
    plt.show()
    


if __name__ == '__main__':
    main()