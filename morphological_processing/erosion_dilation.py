import numpy as np
import matplotlib.pyplot as plt
import cv2


def erosion(image, s_element):
    Ms = s_element.shape[0]
    Ns = s_element.shape[1]
    
    M = image.shape[0]
    N = image.shape[1]
    
    target = np.sum(s_element)
    
    Xp = M + 2 * (Ms // 2)
    Yp = N + 2 * (Ns // 2)
    
    image_p = np.zeros((Xp, Yp))
    image_p[Ms // 2: M + Ms // 2, Ns // 2: N + Ns // 2] = image
    new_image = np.zeros_like(image_p)
    
    
    for i in range(Xp - 2 * (Ms//2)):
        for j in range(Yp - 2 * (Ns//2)):
            x = image_p[i: i + Ms, j : j + Ns] * s_element
            # print(i, i + Ms, j, j + Ns, np.sum(x))
            new_image[i + Ms//2, j + Ns//2] = np.sum(x)
    
    new_image[ new_image != target] = 0
    new_image[ new_image == target] = 1
    # print(new_image)
    # return image_p - new_image
    return new_image[Ms // 2: M + Ms // 2, Ns // 2: N + Ns // 2]


def dilation(image, s_element):
    M = image.shape[0]
    N = image.shape[1]
    
    Ms = s_element.shape[0]
    Ns = s_element.shape[1]

    Xp = M + 2 * (Ms // 2)
    Yp = N + 2 * (Ns // 2)
    
    image_p = np.zeros((Xp, Yp))
    image_p[Ms // 2 : M + Ms//2, Ns // 2 : N + Ns//2] = image
    new_image = np.zeros_like(image_p)
    
    for i in range(Xp - 2 * (Ms//2)):
        for j in range(Yp - 2 * (Ns//2)):
            x = image_p[i : i + Ms, j : j + Ns] * s_element
            new_image[i + Ms//2, j + Ns//2] = np.sum(x)
    
    new_image[new_image > 0] = 1
    return new_image[Ms // 2: M + Ms // 2, Ns // 2: N + Ns // 2]
    

def opening(image, s_element):
    eroted_image = erosion(image, s_element)
    dilated_image = dilation(eroted_image, s_element)
    
    return dilated_image


def closing(image, s_element):
    dilated_image = dilation(image, s_element)
    eroted_image = erosion(dilated_image, s_element)
    
    return eroted_image

def boundary_extraction(image, s_element):
    return image - erosion(image, s_element)


if __name__ == '__main__':
    image = cv2.imread('./images/wirebond-mask.tif', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    
    image[image > 0] = 1
    
    # print(image)
    s_element  = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ])
    
    # s_element  = np.array([
    #     [0, 1, 0],
    #     [1, 1, 1],
    #     [0, 1, 0],
    # ])
    
    eroted_image = erosion(image, s_element)
    dilated_image = dilation(image, s_element)
    
    opened_image = opening(image, s_element)
    closed_image = closing(image, s_element)
    boundary = boundary_extraction(image, s_element)
    # exit()

    plt.figure()
    plt.imshow(eroted_image, cmap='gray')
    plt.title('Eroted Image')
    plt.axis('off')
    
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # plt.show()
    # exit()
    
    plt.figure()
    plt.imshow(dilated_image, cmap='gray')
    plt.title('Dilated Image')
    plt.axis('off')
    
    plt.figure()
    plt.imshow(opened_image, cmap='gray')
    plt.title('Image after Opening')
    plt.axis('off')
    
    plt.figure()
    plt.imshow(closed_image, cmap='gray')
    plt.title('Image after Closing')
    plt.axis('off')
    
    plt.figure()
    plt.imshow(boundary, cmap='gray')
    plt.title('Boundary Image')
    plt.axis('off')
    
    plt.show()
    
    
    