import neural_net
import mnist_loader
from pprint import pprint

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print("loading_done")
net = neural_net.NeuralNet([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)