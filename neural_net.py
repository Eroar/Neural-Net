import math
import random
import numpy

def sigmoid(z):
    """ 
    sigmoid function
    returns a value between 0 and 1
    """
    return 1.0/(1.0 + numpy.exp(-z))

def sigmoidPrime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def tanh(z):
    return numpy.tanh(z)

def tanhPrime(z):
    return 1 - numpy.square(z)

def relu(z):
    return numpy.where(z > 0, z, 0.01*z)

def reluPrime(z):
    return numpy.where(z > 0, 1.0, 0.01)

class NeuralNet():

    def __init__(self, sizes, seed="No seed", debug=False, activationFunc="sigmoid"):
        self.debug = debug
        self.numLayers = len(sizes)
        self.sizes = sizes
        if seed != "No seed":
            if self.debug:
                print("Setting seed")
            numpy.random.seed(seed)
            random.seed(seed)

        self.biases = []
        for neuronsInLayerNum in sizes[1:]:
            self.biases.append(numpy.random.randn(neuronsInLayerNum, 1))

        self.weights = []
        for neuronsInLayerNum, neuronsInPrevLayer in zip(sizes[1:], sizes[:-1]):
            self.weights.append(numpy.random.randn(neuronsInLayerNum, neuronsInPrevLayer))

        if activationFunc=="sigmoid":
            self.activationFunc = sigmoid
            self.activationFuncDerivative = sigmoidPrime

        elif activationFunc=="relu":
            self.activationFunc = relu
            self.activationFuncDerivative = reluPrime

        elif activationFunc=="tanh":
            self.activationFunc = tanh
            self.activationFuncDerivative = tanhPrime

    def feedforward(self, a):
        """
        return the output of the network if "a" is input.
        """
        lastLayerActivations = a
        #loops through the array of arrays weights and arrays of biases
        for b, w in zip(self.biases, self.weights):
            #calculates the output of an layer and stores it in a
            """
            calculates the dot product of weights of each neuron and activations of last layer
            and applies sigmoid function to it.
            Numpy applies sigmoid function automatically when the input is an array
            """
            newActivations = numpy.zeros(b.shape)
            #loop through every neuron in a layer
            neuronIndex = 0
            for neutronBias, neutronWeights in zip(b, w):
                newActivations[neuronIndex] = self.activationFunc(numpy.dot(neutronWeights, lastLayerActivations) + neutronBias)
                neuronIndex += 1
            lastLayerActivations = numpy.copy(newActivations)
        #returns the last layer activations array
        return lastLayerActivations
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        """
        n = len(training_data)
        for epoch_num in range(epochs):
            
            #shuffle the training data
            random.shuffle(training_data)
            miniBatches = []
            for k in range(0, n, mini_batch_size):
                miniBatches.append(training_data[k:k+mini_batch_size])

            for miniBatch in miniBatches:
                self.update_mini_batch(miniBatch, eta)
            if test_data:
                n_test = len(test_data)
                if self.debug:
                    print("Epoch {0}: {1} / {2}".format(epoch_num, self.evaluate(test_data), n_test))
            else:
                if self.debug:
                    print("Epoch {0} complete".format(epoch_num))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""

        #prepares an array for needed changes to biases
        nabla_b = []
        for b in self.biases:
            nabla_b.append(numpy.zeros(b.shape))

        #prepares an array for needed changes to weights
        nabla_w = []
        for w in self.weights:
            nabla_w.append(numpy.zeros(w.shape))

        for image, label in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(image, label)
            #adds delta_nabla_b matrix to nabla_b
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            #adds delta_nabla_w matrix to nabla_w
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        index = 0
        for w, nw in zip(self.weights, nabla_w):
            self.weights[index] = w - (eta/len(mini_batch))*nw
            index += 1
        
        index = 0
        for b, nb in zip(self.biases, nabla_b):
            self.biases[index] = b - (eta/len(mini_batch))*nb
            index += 1
    
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        nabla_b = []
        for b in self.biases:
            nabla_b.append(numpy.zeros(b.shape))
        
        nabla_w = []
        for w in self.weights:
            nabla_w.append(numpy.zeros(w.shape))
        # feedforward
        activation = numpy.array(x).reshape(-1, 1)
        activations = [activation] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        
        #loop through every layer in network
        for layer in range(0, len(self.biases)):
            layerZs = []
            layerActivations = []

            #loops through every neuron in the layer
            for neuronBias, neuronWeights in zip(self.biases[layer], self.weights[layer]):
                z = numpy.dot(activation.transpose(), neuronWeights) + neuronBias.transpose()
                layerZs.append(z)
                neuronActivation = self.activationFunc(z)
                layerActivations.append(neuronActivation)

            zs.append(numpy.array(layerZs))
            activation = numpy.array(layerActivations)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * self.activationFuncDerivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())

        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.
        for l in range(2, self.numLayers):
            z = zs[-l]
            activationFunctionDerivativeOfZ = self.activationFuncDerivative(z)
            delta = numpy.dot(self.weights[-l+1].transpose(), delta) * activationFunctionDerivativeOfZ
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = []
        for (x, y) in test_data:
            test_results.append((numpy.argmax(self.feedforward(x)), numpy.argmax(y)))

        correctCount = 0
        for (x,y) in test_results:
            if x == y:
                correctCount += 1
        return correctCount

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        #DEBUG
        # print("COST derivative:", output_activations-y)
        return output_activations-y