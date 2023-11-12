import numpy as np
import matplotlib.pyplot as plt
import cv2



def get_mask(type):
    masks = {
        'horizontal': np.array([
        [-1, -1, -1],
        [ 2,  2,  2],
        [-1, -1, -1],
    ]),
    
    'vertical': np.array([
        [-1, 2, -1],
        [-1, 2, -1],
        [-1, 2, -1],
    ]),
    
    'p45': np.array([
        [ 2, -1, -1],
        [-1,  2, -1],
        [-1, -1,  2],
    ]),
    
    'n45': np.array([
        [-1, -1,  2],
        [-1,  2, -1],
        [ 2, -1, -1],
    ])
    }

    return masks[type]

def line_detect(image, type='horizontal'):
    mask = get_mask(type)

    M = image.shape[0]
    N = image.shape[1]
    
    pad_x = 1
    pad_y = 1
    
    new_image = np.zeros((M+pad_x, N+pad_y))
    image_p = np.zeros((M+pad_x, N+pad_y))
    image_p[pad_x: M + pad_x, pad_y: N + pad_y] = image

    for i in range(M - 2 * pad_x):
        for j in range(N - 2 * pad_y):
            x = image_p[i : i + mask.shape[0], j : j + mask.shape[1]] * mask
            new_image[i + pad_x, j + pad_y] = np.sum(x)
    
    new_image[new_image <= 0] = 0
    new_image = new_image - np.min(new_image)
    new_image = new_image / np.max(new_image)
    new_image = new_image * 255
    new_image =  new_image[pad_x: M + pad_x, pad_y: N + pad_y].astype(np.uint8)
    # new_image[new_image > 0] = 1
    return new_image



if __name__ == '__main__':
    image = cv2.imread('./images/wirebond-mask.tif', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    
    h_line = line_detect(image, 'horizontal')
    v_line = line_detect(image, 'vertical')
    p45_line = line_detect(image, 'p45')
    n45_line = line_detect(image, 'n45')
    
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.figure()
    plt.imshow(h_line, cmap='gray')
    plt.title('Horizontal Lines')
    plt.axis('off')
    
    plt.figure()
    plt.imshow(v_line, cmap='gray')
    plt.title('Vertical Lines')
    plt.axis('off')
    
    plt.figure()
    plt.imshow(p45_line, cmap='gray')
    plt.title('Pos 45 Lines')
    plt.axis('off')
    
    plt.figure()
    plt.imshow(n45_line, cmap='gray')
    plt.title('Neg 45 Lines')
    plt.axis('off')
    
    plt.show()