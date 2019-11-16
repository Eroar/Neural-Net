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

    listOfScores = list(learningRatesDicts.values())

    index = 0
    for i in range(len(listOfScores)):
        listOfScores[index] = list(map(lambda x: x/100, listOfScores[index]))
        listOfScores[index].insert(0, 0)
        index += 1
        

    for eta, scores in zip(etas, listOfScores):
        plt.plot(scores, label=str(eta), linestyle=(0, (5, 10)))

    plt.title("Graph showing the learning performance of NN with different learning rates")
    plt.legend(loc="best")
    plt.ylabel("Percentage of correctly labeled numbers")
    plt.xlabel("Epoch Number")
    plt.xticks([i for i in range(0, 31, 1)])
    plt.savefig(fname="Learning_Rates_Graph.png")
    plt.show()
