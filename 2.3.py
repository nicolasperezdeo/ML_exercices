import csv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.linalg import hilbert


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

price = columns[8][:]
plt.scatter(columns[0][:],columns[1][:], s=0.1,c=columns[8][:], alpha=0.1)
#plt.colorbar()
plt.show()

np.random.seed(1234)



