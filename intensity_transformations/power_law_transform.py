import cv2
import numpy as np
import matplotlib.pyplot as plt


def power_law(image: np.array, c: float, gamma: float):
    image = image.astype(np.float32) / 255.0
    image =  c * np.power(image, gamma) * 255
    return image.astype(np.uint8)


def get_transform_fn(c:float, gamma:float):
    x = np.array([i for i in range(256)])
    y = power_law(x, c, gamma)
    return x, y


if __name__ == '__main__':
    image = cv2.imread('./images/cat.jpg')
    image = cv2.resize(image, (512, 512))
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    c = 1
    
    i = 1
    gammas = [0.04, 0.5, 1, 3, 5, 25]
    for gamma in gammas:
        transformed_image = power_law(image=gray_image, c=c, gamma=gamma)

        plt.subplot(2,3,i)
        plt.imshow(transformed_image, cmap='gray')
        plt.axis('off')
        plt.title(f'Gamma: {gamma}')
        i+= 1

    plt.figure()
    for gamma in gammas:
        transform_fn = get_transform_fn(c, gamma=gamma)
        plt.plot(transform_fn[0], transform_fn[1])
        plt.axis('on')
        plt.title(f'Transform Functions')
    plt.legend( [f'Gamma: {i}' for i in gammas])
    plt.show()