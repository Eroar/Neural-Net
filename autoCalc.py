import os

batchSizes = [40, 50, 100]

for batchSize in batchSizes:
    os.system(f"main.py -f etas.txt -batchS {batchSize} -sizes 784,30,10 -epochs 30 -seed 2534")
    print("#"*50)
    print("Batch size finished:", batchSize)
    print("#"*50)