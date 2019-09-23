import neural_net
import mnist_loader

trainingData = mnist_loader.loadTrainingData()
# testData = mnist_loader.loadTestData()
print("loading_done")
net = neural_net.NeuralNet([784, 30, 10])
net.SGD(trainingData, 30, 10, 3)#, test_data=test_data)