import numpy as np
import matplotlib.pyplot as plt
import json

import pyperclip
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

# # only for learning rates, through every epoch
# numOfRows = 30 + 1
# numOfColumns = 1+ len(bestScores)

# outArr = [["" for i in range(numOfColumns)] for row in range(numOfRows)]
# #row 1, etas
# for eta, i in zip(bestScores, range(1,11)):
#     outArr[0][i] = str(eta)

# #left column, epochs
# for epochNum in range(1, 31):
#     outArr[epochNum][0] = str(epochNum)

# #data insertion
# fileName = activationFunction + "-" + str(sizes) + "-" + str(30) + ".json"
# with open(fileName, "r") as f:
#     results = json.load(f)
# for eta, etaNum in zip(results, range(len(results))):
#     for epoch in range(30):
#         outArr[1+epoch][1+etaNum] = str(results[eta][epoch]/100)

# outString = ""
# for row in outArr:
#     outString += "\t".join(row)
#     outString += "\n"
# pyperclip.copy(outString)


#BEST RESULTS, learning rate + mini-batch size
numOfRows = 30 + 1
numOfColumns = 1+ len(bestScores)

outArr = [["" for i in range(numOfColumns)] for row in range(numOfRows)]
#row 1, etas
for eta, i in zip(bestScores, range(1,11)):
    outArr[0][i] = str(eta)

#left column, epochs
for i, batchSize in enumerate(miniBatchSizes):
    outArr[i+1][0] = str(batchSize)

#data insertion
for x, eta in enumerate(bestScores):
    print("eta", eta)
    for y, batchSize in enumerate(bestScores[eta]):
        outArr[1+y][1+x] = str(bestScores[eta][batchSize]/100)

outString = ""
for row in outArr:
    outString += "\t".join(row)
    outString += "\n"
pyperclip.copy(outString)


# Plotting point using scatter method
# plt.scatter(X,Y)
# plt.show()