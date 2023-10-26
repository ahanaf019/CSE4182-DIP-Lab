import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_histogram(image: np.array):
    x = np.array([ i for i in range(256) ])
    y = np.array([ 0 for i in range(256) ], dtype=np.int32)
    
    c = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            y[image[i, j]] += 1
            c += 1


    
    return x, y


def equalize_hist(image: np.array):
    L = 256
    x, y = get_histogram(image)
    mn = image.shape[0] * image.shape[1]
    
    pr = y / mn
    print(np.sum(y))
    print(np.sum(pr))
    print(np.min(pr), np.max(pr))
    s = []
    sk = 0
    for i in range(L):
        sk += pr[i]
        s.append(np.round(sk * (L - 1)))
    return x, np.array(s)
    

if __name__ == '__main__':
    image = cv2.imread('./images/cat.jpg')
    image = cv2.resize(image, (512, 512))
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    x, y = get_histogram(gray_image)
    plt.subplot(2, 1, 1)
    plt.bar(x, y)
    plt.axis('on')
    plt.title(f'Histogram')
    
    x, y = equalize_hist(gray_image)
    plt.subplot(2, 1, 2)
    plt.bar(x, y)
    plt.axis('on')
    plt.title(f'Equalized Histogram')
    plt.show()
    
    # c = 1
    
    # i = 1
    # gammas = [0.04, 0.5, 1, 3, 5, 25]
    # for gamma in gammas:
    #     transformed_image = power_law(image=gray_image, c=c, gamma=gamma)

    #     plt.subplot(2,3,i)
    #     plt.imshow(transformed_image, cmap='gray')
    #     plt.axis('off')
    #     plt.title(f'Gamma: {gamma}')
    #     i+= 1

    # plt.figure()
    # for gamma in gammas:
    #     transform_fn = get_transform_fn(c, gamma=gamma)
    #     plt.plot(transform_fn[0], transform_fn[1])
    #     plt.axis('on')
    #     plt.title(f'Transform Functions')
    # plt.legend( [f'Gamma: {i}' for i in gammas])
    # plt.show()