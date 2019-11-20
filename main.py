import neural_net
import mnist_loader
import numpy

#for research
import json
import multiprocessing
import os
import sys

import warnings
warnings.filterwarnings('ignore')

trainingData = mnist_loader.loadTrainingData()
testData = mnist_loader.loadTestData()
print("loading_done")


learningRates = [0.001, 0.05, 0.065, 0.07, 0.075, 0.08, 0.09, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10] #sigmoid

numOfEpochs2Calc = 30
nnSizes = [784, 30, 10]
miniBatchSize = 50
activationFunc = "sigmoid" #"relu" or "sigmoid" or "tanh"
costFunc = "" #cross_entropy
useSoftMax = False
seed = 0
poolSize = None
debug = True
debugWB = True #debug weights and biases
ignoreCalculated = False

results = []

def getResultsJson():
    try:
        with open("results.json", "r") as f:
            resultsDict = json.load(f)
    except:
        # print("No Results.json file found, creating a new one")
        resultsDict = {"relu": {}, "tanh":{}, "sigmoid": {}}
    return resultsDict

def addResults2Json(sizes, eta, results, activationFunc, miniBatchSize):
    resultsDict = getResultsJson()
    try:
        resultsDict[str(activationFunc)][str(miniBatchSize)][str(sizes)][str(eta)] = results
    except KeyError:
        resultsDict[str(activationFunc)][str(miniBatchSize)]= {str(sizes): {str(eta) : results}}
    
    with open("results.json", "w") as f:
        json.dump(resultsDict, f, indent=4)

def addNetworkSettings2Json(weights, biases, eta, activationFunc, sizes, epoch, miniBatchSize):
    cwdPath = os.getcwd()
    folderPath = os.path.join(cwdPath, "NN_settings", activationFunc, str(sizes), str(miniBatchSize), str(eta), str(epoch))
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    weightsToSave = {}
    for layer,layerIndex in zip(weights, range(len(weights))):
        weightsToSave["Layer "+ str(layerIndex)] = {}
        for neuron, neuronIndex in zip(layer, range(len(layer))):
            weightsToSave["Layer "+ str(layerIndex)]["Neuron " + str(neuronIndex)] = neuron.tolist()

    biasesToSave = {}
    for layer,layerIndex in zip(biases, range(len(biases))):
        biasesToSave["Layer "+ str(layerIndex)] = {}
        for neuron, neuronIndex in zip(layer, range(len(layer))):
            biasesToSave["Layer "+ str(layerIndex)]["Neuron " + str(neuronIndex)] = neuron.tolist()

    with open(os.path.join(folderPath, "weights") + ".json", "w") as f:
        json.dump(weightsToSave, f, indent=4)    
    
    with open(os.path.join(folderPath, "biases") + ".json", "w") as f:
        json.dump(biasesToSave, f, indent=4)

def performForLearningRate(eta):
    epochsPerformance = []
    
    print("START : Starting to train, eta:", eta)
    net = neural_net.NeuralNet(nnSizes, seed=seed, debug=False, activationFunc=activationFunc, costFunc=costFunc, useSoftMax=useSoftMax)
    for epoch in range(numOfEpochs2Calc):
        net.SGD(trainingData, 1, mini_batch_size=miniBatchSize, eta=eta)
        performance = net.evaluate(testData)
        if debug:
            print("PERFORMANCE : Learning rate:", eta, "epoch:", str(epoch+1), "performance:", performance)
        epochsPerformance.append(performance)
        if debugWB:
            addNetworkSettings2Json(net.weights, net.biases, eta, activationFunc, nnSizes, epoch, miniBatchSize)
        print("EPOCH : Learning rate:", eta, "epoch:", str(epoch+1), "finished")

    print("FINISH : Learning rate:", eta, "finished")

    return epochsPerformance

def getEtasToCalculate(ignoreCalculated):
    if ignoreCalculated:
        return learningRates
    try:
        calculatedEtasStrings = list(getResultsJson()[str(activationFunc)][str(nnSizes)].keys())
    except KeyError:
        calculatedEtasStrings = []

    calculatedLearningRates = set(map(float, calculatedEtasStrings))
    toCalculate = [eta for eta in learningRates if eta not in calculatedLearningRates]
    return toCalculate

if __name__=="__main__":
    print("Activation function:", activationFunc)    
    if poolSize:
        pool = multiprocessing.Pool(poolSize)
    else:
        pool = multiprocessing.Pool()

    toCalculate = getEtasToCalculate(ignoreCalculated)
    print("Learning rates to compute:", toCalculate)
    print("Number of rates to compute", len(toCalculate))


    # performForLearningRate(toCalculate[0])
    learningRatesResults = pool.map(performForLearningRate, toCalculate)

    for i in range(len(toCalculate)):
        addResults2Json(nnSizes, toCalculate[i], learningRatesResults[i], activationFunc, miniBatchSize)
