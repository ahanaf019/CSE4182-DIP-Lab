import cv2
import matplotlib.pylab as plt

def create_patches(image, patch_per_axis):
    patch_size_x = image.shape[0] // patch_per_axis
    patch_size_y = image.shape[1] // patch_per_axis
    
    plt.figure(figsize=(10,10))
    patches = []
    c = 1
    for i in range(patch_per_axis):
        
        x_start = i * patch_size_x
        x_end = (i + 1) * patch_size_x
        
        for j in range(patch_per_axis):
            y_start = j * patch_size_y
            y_end = (j + 1) * patch_size_y
            # print(x_start, x_end)
            img = image[x_start : x_end, y_start : y_end, :]
            plt.subplot(patch_per_axis, patch_per_axis, c)
            plt.imshow(img)
            plt.axis('off')
            c += 1
            # break
    plt.show()


if __name__ == "__main__":
    image = cv2.imread('image.jpg')
    image = cv2.resize(image, (512,512))
    create_patches(image, 16)