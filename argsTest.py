import sys

poolSizeLimit = None
etasFileName = "" #required
miniBatchSize = None #required
seed = None #required for research
debug = False
debugWB = False
ignoreCalculated = False
outputFileName = "" #required
nnSizes = None

args = sys.argv[1:]

if "-p" not in args:
    print("No p of f")

for arg, i in zip(args, range(len(args))):
    if arg == "-p":
        poolSizeLimit = args[i+1]
    elif arg == "-f":
        etasFileName = args[i+1]
    elif arg == "-b":
        miniBatchSize = args[i+1]
    elif arg == "-s":
        seed = args[i+1]
    elif arg == "-d":
        debug = True
    elif arg == "-dWB":
        debugWB = True
    elif arg == "-i":
        ignoreCalculated = True
    elif arg == "-o":
        outputFileName = args[i+1]
    elif arg == "-sizes":
        nnSizes = args[i+1]

sizes = nnSizes.split(",")
sizes = list(map(int, sizes))
print(sizes)    


# print("poolSizeLimit:", poolSizeLimit)
# print("etasFileName:", etasFileName)
# print("batchSize:", miniBatchSize)
# print("seed:", seed)
# print("debug:", debug)
# print("debugWB:", debugWB)
# print("ignoreCalculated:", ignoreCalculated)
# print("outputFileName:", outputFileName)
