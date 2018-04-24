import csv
import numpy as np
import pandas as pd
import random
import math
import operator
import matplotlib.pyplot as plt
from scipy.linalg import hilbert
random.seed(1234)

def L2(data):
    S = np.sum((data-np.mean(data)^2))
    return S


def loadDataSet(fileName):
    with open(fileName) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        data = [r for r in reader]

    return data


def split_dataset(data, threshold):
    data_size = np.size(data, axis=0)
    training_size = int(threshold * data_size)
    test_size = int(np.size(price) - training_size)

    indexes = random.sample(range(data_size), data_size)
    train_indexes = indexes[:training_size]
    test_indexes = indexes[training_size:data_size]

    trainSet = []
    testSet = []

    for i in train_indexes:
        trainSet.append([float(data[i][0]), float(data[i][1]), (data[i][-1])])
        # trainSet.append(data[i])

    for i in test_indexes:
        testSet.append([float(data[i][0]), float(data[i][1]), (data[i][-1])])
        # testSet.append(data[i])

    return trainSet, testSet

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
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


data = loadDataSet('housing.csv')
columns = list(map(list, zip(*data)))

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

plt.scatter(columns[0][:],columns[1][:], s=0.1,c=norm_price, alpha=0.1)
plt.colorbar()
plt.show()

#indexs = np.random.choice(np.arange(tr_size),size=np.arange(tr_size),replace=False)

trainSet, testSet = split_dataset(data,0.8)

testSet2 = testSet[1:100]
#testInstance = [125, 50, 100000]


k = 10
#neighbors = getNeighbors(trainSet, testInstance, k)
#response = getResponse(neighbors)

#print(neighbors)
#print(response)

print('Train set: ' + repr(len(trainSet)))
predictions=[]
for x in range(len(testSet2)):
    neighbors = getNeighbors(trainSet, testSet2[x], k)
    result = getResponse(neighbors)
    predictions.append(result)
    print('> predicted=' + repr(result) + ', actual=' + repr(testSet2[x][-1]))
accuracy = getAccuracy(testSet2, predictions)
print('Accuracy: ' + repr(accuracy) + '%')





