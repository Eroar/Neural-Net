import gzip
import numpy

def loadTrainingLabels():
    labels= numpy.zeros((60000,), dtype=int)
    with gzip.open("MNIST/train-labels-idx1-ubyte.gz", "r") as f: 
        f.read(8)


        for i in range(60000):   
            buffer = f.read(1)
            label = numpy.frombuffer(buffer, dtype=numpy.uint8).astype(numpy.uint8)
            labels[i] = label
    return labels

def loadTrainingImages():
    images = numpy.zeros((60000,784), dtype=numpy.uint8)
    with gzip.open('MNIST/train-images-idx3-ubyte.gz','r') as f:

        image_size = 28

        f.read(16)
        for imageNum in range(60000):
            buffer = f.read(image_size * image_size)
            image = numpy.frombuffer(buffer, dtype=numpy.uint8)
            images[imageNum] = image
    return images

def loadTrainingData():
    """
    returns an array of tupples, at first position of the tuple 
    is the image array and the second one is label
    """
    images = loadTrainingImages()
    labels = loadTrainingLabels()

    data = []

    for (image, label) in zip(images, labels):
        data.append((image, label))

    return data

if __name__=="__main__":
    trainingLabels = loadTrainingLabels()
    print(trainingLabels[59999])

    trainingImages = loadTrainingImages()
    print(trainingImages[59999])

    loadTrainingData()