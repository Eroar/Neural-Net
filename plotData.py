from matplotlib import pyplot as plt
import json

def getResultsJson():
    try:
        with open("results.json", "r") as f:
            resultsDict = json.load(f)
    except:
        resultsDict = {}
        print("No results.json file found")
        raise
    return resultsDict

if __name__=="__main__":
    resultsDict = getResultsJson()
    learningRatesDicts = resultsDict["sigmoid"]["[784, 30, 10]"]
    etas = learningRatesDicts.keys()
    listOfScores = learningRatesDicts.values()

    for eta, scores in zip(etas, listOfScores):
        plt.plot(scores, label=str(eta))

    plt.title("Graph showing the learning performance of NN with different learning rates")
    plt.legend(loc="best")
    plt.ylabel("Correctly labeled numbers")
    plt.xlabel("Epoch Number")
    plt.xticks([i for i in range(0, 32, 2)])
    plt.savefig(fname="Learning_Rates_Graph.png")
    plt.show()
