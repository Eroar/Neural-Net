import neural_net
import mnist_loader
import numpy

#for research
import json

import warnings
warnings.filterwarnings('ignore')

trainingData = mnist_loader.loadTrainingData()
testData = mnist_loader.loadTestData()
print("loading_done")
# net = neural_net.NeuralNet([784, 30, 10], seed=0)
# net.SGD(trainingData, 30, 10, 0.1, test_data=testData)


learningRates = [0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.09, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5]
# learningRates = [0.01, 0.03]
results = []

try:
    with open("results.json", "r") as f:
        resultsDict = json.load(f)
        print(resultsDict)
except:
    print("No Results.json file found, creating a new one")
    resultsDict = {}

for eta in learningRates:
    if str(eta) not in resultsDict:
        print("Starting to train, eta:", eta)
        net = neural_net.NeuralNet([784, 30, 10], seed=0)
        net.SGD(trainingData, 30, 10, eta)
        results.append(net.evaluate(testData))
        print("Learning rate:", eta, "result:", results[-1])
        resultsDict[eta] = results[-1]

    with open("results.json", "w") as f:
        json.dump(resultsDict, f, indent=4)
