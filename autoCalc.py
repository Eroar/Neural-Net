import os

batchSizes = [10, 20, 30]

for batchSize in batchSizes:
    os.system(f"main.py -f etas.txt -batchS {batchSize} -sizes 784,30,10 -epochs 30 -seed 2534 -p 3")
    print("#"*50)
    print("Batch size finished:", batchSize)
    print("#"*50)