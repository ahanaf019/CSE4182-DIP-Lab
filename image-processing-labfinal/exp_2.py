import numpy as np
import matplotlib.pyplot as plt
import cv2


def power_law(image, gamma=1):
    new_image = image.astype(np.float32) / 255
    new_image =  1 * np.power(new_image, gamma)
    new_image =  new_image * 255
    return new_image.astype(np.uint8)


def inverse_log(image):
    c = 255 / np.log(256)
    new_image = image.astype(np.float32)
    new_image =  np.exp(new_image / c) - 1
    new_image =  new_image / np.max(new_image)
    new_image =  new_image * 255
    return new_image.astype(np.uint8)

def enhance_brightness(image: np.ndarray, inc=0, rng=(0, 255)):
    print(image.shape[0])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] >=rng[0] and image[i, j] <= rng[1]:
                image[i, j] += inc
    return image 

def histogram(image: np.ndarray):
    histogram = np.zeros((256))
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            histogram[image[i, j]] += 1
    return histogram

def msb_image(image, bits=3):
    mask = 0b11100000
    return np.bitwise_and(image, mask)


def main():
    image = cv2.imread('./images/cat.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    line = np.linspace(0, 255, 256)
    # print(line)
    # plt.figure()
    # plt.subplot(3,4,1)
    # plt.imshow(image, cmap='gray')
    # plt.subplot(3,4,2)
    # plt.imshow((power_law(image, gamma=0.1)), cmap='gray')
    # plt.subplot(3,4,3)
    # plt.imshow((power_law(image, gamma=0.5)), cmap='gray')
    # plt.subplot(3,4,4)
    # plt.imshow((power_law(image, gamma=1)), cmap='gray')
    # plt.subplot(3,4,5)
    # plt.imshow((power_law(image, gamma=2)), cmap='gray')
    # plt.subplot(3,4,6)
    # plt.imshow((power_law(image, gamma=5)), cmap='gray')
    # plt.subplot(3,4,7)
    # plt.imshow((power_law(image, gamma=10)), cmap='gray')
    # plt.subplot(3,4,8)
    # plt.imshow((power_law(image, gamma=25)), cmap='gray')
    # plt.subplot(3,1,3)
    # plt.plot((power_law(line, gamma=0.1)))
    # plt.plot((power_law(line, gamma=0.5)))
    # plt.plot((power_law(line, gamma=1)))
    # plt.plot((power_law(line, gamma=2)))
    # plt.plot((power_law(line, gamma=5)))
    # plt.plot((power_law(line, gamma=10)))
    # plt.plot((power_law(line, gamma=25)))
    # plt.show()
    
    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.imshow(image, cmap='gray')
    # plt.subplot(2,2,2)
    # plt.imshow((inverse_log(image)), cmap='gray')
    # plt.subplot(2,1,2)
    # plt.plot((inverse_log(line)))
    # plt.show()
    
    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.imshow(image, cmap='gray')
    # plt.subplot(2,2,2)
    # plt.imshow(enhance_brightness(image, 40, (80, 130)), cmap='gray')
    # plt.subplot(2,1,2)
    # plt.plot(histogram(enhance_brightness(image, 40, (80, 130))))
    # plt.show()
    
    msb = msb_image(image, bits=3)
    diff = image - msb
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(image, cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(msb, cmap='gray')
    plt.subplot(2,1,2)
    plt.imshow(diff, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()