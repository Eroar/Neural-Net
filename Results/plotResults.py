import numpy as np
import matplotlib.pyplot as plt
import json
from pprint import pprint

activationFunction = "sigmoid"
sizes = [784, 30, 10]
miniBatchSizes = [10, 20, 30, 40, 50, 100]

bestScores = {}
fileName = activationFunction + "-" + str(sizes) + "-" + str(miniBatchSizes[0]) + ".json"
with open(fileName, "r") as f:
    results = json.load(f)
    for key in results:
        bestScores[key] = {}

for batchSize in miniBatchSizes:
    fileName = activationFunction + "-" + str(sizes) + "-" + str(batchSize) + ".json"
    with open(fileName, "r") as f:
        results = json.load(f)
    for key in results:
        bestScores[key][batchSize] = max(results[key])


for eta in bestScores:
    print(eta)
    for batchSize in bestScores[eta]:
        print(" "*4,batchSize, ":", bestScores[eta][batchSize])

# Plotting point using scatter method
# plt.scatter(X,Y)
# plt.show()