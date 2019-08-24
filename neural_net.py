import math
import random
import numpy

def sigmoid(z):
    """ 
    sigmoid function
    returns a value between 0 and 1
    """
    return 1.0/(1.0 + numpy.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

class NeuralNet():

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [numpy.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [numpy.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        for w in self.weights:
            print("weight shape: ", w.shape)

    def feedforward(self, a):
        """
        return the output of the network if "a" is input.
        """
        #loops through the array of arrays weights and arrays of biases
        for w, b in zip(self.biases, self.weights):
            #calculates the output of an layer and stores it in a
            """
            calculates the dot product of weights of each neuron and activations of last layer
            and applies sigmoid function to it.
            Numpy applies sigmoid function automatically when the input is an array
            """
            print("a len:", a.shape)
            a = sigmoid(numpy.dot(w, a)+b)
            print("a: ", a)
        #returns the last layer activations array
        return a
    
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
        if test_data: 
            n_test = len(test_data)
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
                print ("Epoch {0}: {1} / {2}".format(epoch_num, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(epoch_num))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""

        nabla_b = []
        for b in self.biases:
            nabla_b.append(numpy.zeros(b.shape))

        nabla_w = []
        for w in self.weights:
            nabla_w.append(numpy.zeros(w.shape))

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #adds nabla_b matrix to delta_nabla_b
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            #adds nabla_w matrix to delta_nabla_w
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        #creates an array of zeros
        nabla_b = []
        for b in self.biases:
            nabla_b.append(numpy.zeros(b.shape))

        #creates an array of zeros
        nabla_w = []
        for w in self.weights:
            nabla_w.append(numpy.zeros(w.shape))

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = numpy.dot(self.weights[-l+1].transpose(), delta) * sp
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
            print("x:",len(x))
            test_results.append((numpy.argmax(self.feedforward(x)), y))

        correctCount = 0
        for (x,y) in test_results:
            if x == y:
                correctCount += 1
        return correctCount

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activations-y