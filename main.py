from model import Model
from time import sleep
from datasets.dataset import load_image
import matplotlib.pyplot as plt
import cv2
import numpy as np

PATH = 'datasets'
query = False
search = True

def main():
    test = 0
    try:
        while True:
            image = load_image('test.jpg', size=(224,224))
            # plt.imshow(image)
            # plt.show()
            test += 1
            print(test)
            sleep(2)
    except KeyboardInterrupt:
        print('Bye!')

main()