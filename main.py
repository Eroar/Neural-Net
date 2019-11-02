import neural_net
import mnist_loader
import numpy

#for research
import json
import multiprocessing

import warnings
warnings.filterwarnings('ignore')

trainingData = mnist_loader.loadTrainingData()
testData = mnist_loader.loadTestData()
print("loading_done")
# net = neural_net.NeuralNet([784, 30, 10], seed=0)
# net.SGD(trainingData, 30, 10, 0.1, test_data=testData)


# learningRates = [10, 2, 3, 5]#[0.01, 0.05, 0.065, 0.07, 0.075, 0.08, 0.09, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10]#[0.005, 0.006, 0.007, 0.008, 0.0085, 0.009,0.01, 0.03, 0.05, 0.08, 0.09, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5]
learningRates = [0.00000001, 0.00000005, 0.00000012]#relu

numOfEpochs2Calc = 30
nnSizes = [784, 30, 10]
activationFunc = "relu" #"relu" or "sigmoid"
seed = 0
poolSize = 3
debug = True

results = []

def getResultsJson():
    try:
        with open("results.json", "r") as f:
            resultsDict = json.load(f)
            # print(resultsDict)
    except:
        # print("No Results.json file found, creating a new one")
        resultsDict = {"relu": {}, "sigmoid": {}}
    return resultsDict

def addResults2Json(sizes, eta, results, activationFunc):
    resultsDict = getResultsJson()
    try:
        resultsDict[str(activationFunc)][str(sizes)][str(eta)] = results
    except KeyError:
        resultsDict[str(activationFunc)][str(sizes)] = {str(eta) : results}
    
    with open("results.json", "w") as f:
        json.dump(resultsDict, f, indent=4)

def performForLearningRate(eta):
    epochsPerformance = []
    
    print("Starting to train, eta:", eta)
    net = neural_net.NeuralNet(nnSizes, seed=seed, debug=False, activationFunc=activationFunc)
    for epoch in range(numOfEpochs2Calc):
        net.SGD(trainingData, 1, 10, eta)
        performance = net.evaluate(testData)
        if debug:
            print("Learning rate:", eta, "epoch:", str(epoch+1), "performance:", performance)
        epochsPerformance.append(performance)
        print("Learning rate:", eta, "epoch:", str(epoch+1), "finished")

    print("Learning rate:", eta, "finished")

    return epochsPerformance

def getEtasToCalculate():
    try:
        calculatedEtasStrings = list(getResultsJson()[str(activationFunc)][str(nnSizes)].keys())
    except KeyError:
        calculatedEtasStrings = []

    calculatedLearningRates = set(map(float, calculatedEtasStrings))
    toCalculate = [eta for eta in learningRates if eta not in calculatedLearningRates]
    return toCalculate

if __name__=="__main__":
    pool = multiprocessing.Pool(poolSize)

    toCalculate = getEtasToCalculate()
    print("Learning rates to calculate:", toCalculate)

    #test
    # performForLearningRate(0.0000008)

    learningRatesResults = pool.map(performForLearningRate, toCalculate)

    for i in range(len(toCalculate)):
        addResults2Json(nnSizes, toCalculate[i], learningRatesResults[i], activationFunc)
