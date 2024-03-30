import numpy as np
import cv2

def visualize(size, population, name = 'image'):
    new_img = np.ones(shape=size, dtype=np.uint8)*255
    for individ in population:
        for pixel in individ.pixels:
            new_img[pixel[0], pixel[1]] = 0

    cv2.imshow(name, new_img)
    cv2.waitKey(0)

