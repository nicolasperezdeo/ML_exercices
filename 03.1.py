import csv
import numpy as np
import pandas as pd
import random
import math
import operator
import matplotlib.pyplot as plt
from scipy.linalg import hilbert
random.seed(1234)


a = np.random.uniform([0,100,100])
b = np.random.uniform([0,100,1000])
c = np.random.uniform([0,100,10000])
d = np.random.uniform([0,100,100000])

h1 = plt.hist(a,bins=10)
plt.show()
h2 = plt.hist(b,bins=10)
plt.show()
h3 = plt.hist(c,bins=10)
plt.show()
plt.hist(d,bins=10)
plt.show()