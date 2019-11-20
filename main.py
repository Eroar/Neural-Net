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


poolSizeLimit = None
etasFilename = "" #required
miniBatchSize = None #required
seed = None #required for research
debug = False
debugWB = False
ignoreCalculated = False
outputFilename = "" 
nnSizes = None #required
numOfEpochs2Calc = 0 #required

#NOT IMPLEMENTED
##################
activationFunc = "sigmoid" #"relu" or "sigmoid" or "tanh"
costFunc = "" #cross_entropy
useSoftMax = False
###################


def getResultsJson(outputFilename):
    cwd = os.getcwd()
    folderPath = os.path.join(cwd, "Results")
    if not os.path.isdir(folderPath):
        os.makedirs(folderPath)

    try:
        with open("Results/" + outputFilename, "r") as f:
            resultsDict = json.load(f)
    except:
        # print("No Results.json file found, creating a new one")
        resultsDict = {}
    return resultsDict

def addResults2Json(results, eta, outputFilename):
    resultsDict = getResultsJson(outputFilename)

    resultsDict[eta] = results
    
    with open("Results/" + outputFilename, "w") as f:
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
    addResults2Json(epochsPerformance, eta, outputFilename)
    return epochsPerformance

def filterEtas(etas, outputFilename, ignoreCalculated):
    if ignoreCalculated:
        return etas
    try:
        calculatedEtasStrings = list(getResultsJson(outputFilename).keys())
    except KeyError:
        calculatedEtasStrings = []

    calculatedLearningRates = set(map(float, calculatedEtasStrings))
    toCalculate = [eta for eta in etas if eta not in calculatedLearningRates]
    return toCalculate

if __name__=="__main__":
    args = sys.argv[1:]


    for arg, i in zip(args, range(len(args))):
        if arg == "-p":
            poolSizeLimit = int(args[i+1])
        elif arg == "-f": #required
            etasFilename = args[i+1]
        elif arg == "-batchS": #required
            miniBatchSize = int(args[i+1])
        elif arg == "-seed": #required
            seed = int(args[i+1])
        elif arg == "-d":
            debug = True
        elif arg == "-dWB":
            debugWB = True
        elif arg == "-i":
            ignoreCalculated = True
        elif arg == "-o":
            outputFilename = args[i+1]
        elif arg == "-sizes": #required
            nnSizes = args[i+1]
            nnSizes = nnSizes.split(",")
            nnSizes = list(map(int, nnSizes))
        elif arg == "-epochs": #required
            numOfEpochs2Calc = int(args[i+1])
        elif arg == "-help":
            with open("args_help_file.txt", "r") as f:
                print(f.read())
            quit()

    if debug:
        print("poolSizeLimit:", poolSizeLimit)
        print("etasFilename:", etasFilename)
        print("miniBatchSize:", miniBatchSize)
        print("seed:", seed)
        print("debug:", debug)
        print("debugWB:", debugWB)
        print("ignoreCalculated:", ignoreCalculated)
        print("outputFilename:", outputFilename)
        print("nnSizes:", nnSizes)
        print("numOfEpochs2Calc:", numOfEpochs2Calc)

    quitVar = False
    if "-f" not in args:
        print("-f filename : etas filename not specified")
        quitVar = True
    if "-batchS" not in args:
        print("-batchS int : mini batch size not specified")
        quitVar = True
    if "-sizes" not in args:
        print("-sizes sizes : sizes of network not specified")
        quitVar = True
    if "-epochs" not in args:
        print("-epochs int : number of epochs not specified")
        quitVar = True
    if quitVar:
        quit()

    if outputFilename == "":
        outputFilename = activationFunc + "-" + str(nnSizes) + "-" + str(miniBatchSize) + ".json"

    if poolSizeLimit==1:
        print("Cpu count: 1")
    elif poolSizeLimit:
        pool = multiprocessing.Pool(poolSizeLimit)
        if poolSizeLimit < multiprocessing.cpu_count():
            print("Cpu count:", poolSizeLimit)
        else:
            print("Cpu count:", multiprocessing.cpu_count())
    else:
        pool = multiprocessing.Pool()
        print("Cpu count:", multiprocessing.cpu_count())

    with open(etasFilename, "r") as f:
        etas = f.read().split("\n")
    etas = list(map(float, etas))

    toCalculate = filterEtas(etas, outputFilename, ignoreCalculated)

    print("Activation function:", activationFunc)    
    print("Learning rates to compute:", toCalculate)
    print("Number of learning rates to compute", len(toCalculate))

    tmpDict = {
        "miniBatchSize": miniBatchSize,
        "seed": seed,
        "debug": debug,
        "debugWB": debugWB,
        "outputFilename": outputFilename,
        "nnSizes": nnSizes,
        "numOfEpochs2Calc": numOfEpochs2Calc
    }

    with open("tmp", "w") as f:
        json.dump(tmpDict, f, indent=4)

    if poolSizeLimit == 1:
        learningRatesResults = [performForLearningRate(lr) for lr in toCalculate]
        print("finished")
    else:
        learningRatesResults = pool.map(performForLearningRate, toCalculate)

    cwd = os.getcwd()
    tmpPath = os.path.join(cwd, "tmp")
    if os.path.isfile(tmpPath):
        os.remove(tmpPath)

else:
    with open("tmp", "r") as f:
        tmpDict = json.load(f)

    miniBatchSize = tmpDict["miniBatchSize"]
    seed = tmpDict["seed"]
    debug = tmpDict["debug"]
    debugWB = tmpDict["debugWB"]
    outputFilename = tmpDict["outputFilename"]
    nnSizes = tmpDict["nnSizes"]
    numOfEpochs2Calc = tmpDict["numOfEpochs2Calc"]


    
