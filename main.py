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


learningRates = [0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.09, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5]
numOfEpochs2Calc = 30
nnSizes = [784, 30, 10]
seed = 0
results = []

def getResultsJson():
    try:
        with open("results.json", "r") as f:
            resultsDict = json.load(f)
            # print(resultsDict)
    except:
        # print("No Results.json file found, creating a new one")
        resultsDict = {}
    return resultsDict

def addResults2Json(sizes, eta, results):
    resultsDict = getResultsJson()
    try:
        resultsDict[str(sizes)][str(eta)] = results
    except KeyError:
        resultsDict[str(sizes)] = {str(eta) : results}
    
    with open("results.json", "w") as f:
        json.dump(resultsDict, f, indent=4)

def performForLearningRate(eta):
    epochsPerformance = []
    
    print("Starting to train, eta:", eta)
    net = neural_net.NeuralNet(nnSizes, seed=seed)
    for epoch in range(numOfEpochs2Calc):
        net.SGD(trainingData, 1, 10, eta)
        performance = net.evaluate(testData)
        epochsPerformance.append(performance)
        print("Learning rate:", eta, "epoch:", str(epoch+1), "finished")

    print("Learning rate:", eta, "finished")

    return epochsPerformance

if __name__=="__main__":
    pool = multiprocessing.Pool()

    calculatedLearningRates = set(map(float, getResultsJson()[str(nnSizes)].keys()))
    toCalculate = [eta for eta in learningRates if eta not in calculatedLearningRates]

    learningRatesResults = pool.map(performForLearningRate, toCalculate)
    
    for i in range(len(learningRates)):
        addResults2Json(nnSizes, learningRates[i], learningRatesResults[i])
