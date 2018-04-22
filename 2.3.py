import csv
import numpy as np
import pandas as pd
import random
import math
import operator
import matplotlib.pyplot as plt
from scipy.linalg import hilbert
np.random.seed(1234)


def L2(data):
    S = np.sum((data-np.mean(data)^2))
    return S

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

import operator
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]



def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] is predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

with open("housing.csv") as f:
    reader = csv.reader(f)
    next(reader) # skip header
    data = [r for r in reader]




columns = list(map(list, zip(*data)))
datap = pd.read_csv("housing.csv")
#datap.latitude

dcol = np.size(data,1)
maxs = np.argmax(data,axis=0)
print(maxs)
for i in np.arange(dcol):
    print(data[int(maxs[i])][i])

#norm_col = []

price = (columns[8][:])
#price_i = [float(l[0]) for l in price]
for i in np.arange(np.size(price)):
    price[i] = float(price[i])


minprice = np.min(price,axis=0)
maxprice = np.max(price,axis=0)
norm_col = []

dif = maxprice - minprice

norm_price = []
for i in np.arange(np.size(price,axis=0)):
    norm_price.append((price[i] - minprice) / dif)

#plt.scatter(columns[0][:],columns[1][:], s=0.1,c=norm_price, alpha=0.1)
#plt.colorbar()
#plt.show()

tr_size = int(0.8 * np.size(price))
te_size = int(np.size(price)-tr_size)
data_size = np.size(data,axis=0)

#indexs = np.random.choice(np.arange(tr_size),size=np.arange(tr_size),replace=False)

indexs = random.sample(range(data_size), data_size)
tr_index = indexs[1:tr_size]
te_index = indexs[tr_size+1:data_size]
a = []

trainSet = []
testSet = []
for i in tr_index:
    trainSet.append([float(data[i][0]), float(data[i][1]),float(data[i][8])])

for i in te_index:
    testSet.append([float(data[i][0]), float(data[i][1]),float(data[i][8])])

testInstance = [125, 50, 100000]


k = 100
neighbors = getNeighbors(trainSet, testInstance, k)
response = getResponse(neighbors)

#print(neighbors)
#print(response)

print('Train set: ' + repr(len(trainSet)))
predictions=[]
for x in range(len(testSet)):
    neighbors = getNeighbors(trainSet, testSet[x], k)
    result = getResponse(neighbors)
    predictions.append(result)
    print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')





