import cv2
import numpy as np
import matplotlib.pyplot as plt


def log_transform(image: np.array):
    c = 255 / np.log(256)
    image = image.astype(np.float32) 
    image =  c * np.log(image + 1)
    image = image / np.max(image) * 255
    return image.astype(np.uint8)


def inverse_log_transform(image: np.array):    
    c = 255 / np.log(256)
    image = image.astype(np.float64)
    image =  np.exp(image* (1/c)) - 1
    image = image / np.max(image) * 255
    return image.astype(np.uint8)


def get_transform_fn(transform):
    
    x = np.array([i for i in range(256)])
    y = transform(x)
    return x, y


if __name__ == '__main__':
    image = cv2.imread('./images/cat.jpg')
    image = cv2.resize(image, (512, 512))
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    c = 0.1
    

    transformed_image = log_transform(image=gray_image)
    plt.subplot(2,1,1)
    plt.imshow(transformed_image, cmap='gray')
    plt.axis('off')
    plt.title(f'Log Transform')

    transform_fn = get_transform_fn(log_transform)
    plt.subplot(2,1,2)
    plt.plot(transform_fn[0], transform_fn[1])
    plt.axis('on')
    plt.title(f'Log Transform Function')
    
    plt.figure()
    transformed_image = inverse_log_transform(image=gray_image)
    plt.subplot(2,1,1)
    plt.imshow(transformed_image, cmap='gray')
    plt.axis('off')
    plt.title(f'Inverse Log Transform')

    transform_fn = get_transform_fn(inverse_log_transform)
    plt.subplot(2,1,2)
    plt.plot(transform_fn[0], transform_fn[1])
    plt.axis('on')
    plt.title(f'Transform Function')
    plt.show()