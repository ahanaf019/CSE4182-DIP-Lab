import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    image = cv2.imread('./images/wirebond-mask.tif', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    image = image > 0
    image = image.astype(np.uint8)
    print(image.shape)
    
    s_element = np.array(
        [[0,1,0],
        [1,1,1],
        [0,1,0],]
    )
    
    eroted_image = erosion(image, s_element)
    dilated_image = dilation(image, s_element)
    open_image = opening(image, s_element)
    closed_image = closing(image, s_element)
    boundary_image = boundary_extraction(image, s_element)
    
    plt.subplot(2,3,1)
    plt.imshow(image, cmap='gray')
    
    plt.subplot(2,3,2)
    plt.imshow(eroted_image, cmap='gray')

    plt.subplot(2,3,3)
    plt.imshow(dilated_image, cmap='gray')
    
    plt.subplot(2,3,4)
    plt.imshow(open_image, cmap='gray')
    
    plt.subplot(2,3,5)
    plt.imshow(closed_image, cmap='gray')
    
    plt.subplot(2,3,6)
    plt.imshow(boundary_image, cmap='gray')
    
    plt.show()
    
    
def boundary_extraction(image: np.ndarray, s_element: np.ndarray):
    return image - erosion(image, s_element)

def opening(image: np.ndarray, s_element: np.ndarray):
    x = erosion(image, s_element)
    return dilation(image, s_element)

def closing(image: np.ndarray, s_element: np.ndarray):
    x = dilation(image, s_element)
    return erosion(image, s_element)

    
def erosion(image: np.ndarray, s_element: np.ndarray):
    padx = s_element.shape[0] // 2
    pady = s_element.shape[1] // 2
    
    padded_image = np.zeros((image.shape[0] + 2 * padx, image.shape[1] + 2 * pady))
    padded_image[padx: padx + image.shape[0], pady: pady + image.shape[1]] = image
    new_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x = padded_image[i : i + s_element.shape[0], j : j + s_element.shape[1]] * s_element
            # print(x)
            x = np.sum(x)
            if x == np.sum(s_element):
                new_image[i, j] = 1
    return new_image


def dilation(image: np.ndarray, s_element: np.ndarray):
    padx = s_element.shape[0] // 2
    pady = s_element.shape[1] // 2
    
    padded_image = np.zeros((image.shape[0] + 2 * padx, image.shape[1] + 2 * pady))
    padded_image[padx: padx + image.shape[0], pady: pady + image.shape[1]] = image
    new_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x = padded_image[i : i + s_element.shape[0], j : j + s_element.shape[1]] * s_element
            # print(x)
            x = np.sum(x)
            if x > 0:
                new_image[i, j] = 1
    return new_image


if __name__ == '__main__':
    main()