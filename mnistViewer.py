import pygame
import numpy as np
import mnist_loader
import time


if __name__=="__main__":
    images = mnist_loader.loadTrainingImages()
    labels = mnist_loader.loadTrainingLabels()

    dataNum=0
    while True:
        image = images[dataNum]
        label = labels[dataNum]
        print("labels", labels[0:10])

        print("label:", label)
        import matplotlib.pyplot as plt
        matImage = np.asarray(image.reshape(28,28)).squeeze()
        plt.imshow(matImage)
        plt.draw()
        # plt.show()
        dataNum += 1
        plt.pause(2)