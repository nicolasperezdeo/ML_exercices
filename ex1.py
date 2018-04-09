import numpy as np
import matplotlib.pyplot as plt

from numpy import genfromtxt
my_data = genfromtxt('traffic_per_hour.csv', delimiter='\t')

my_data[np.isnan(my_data)] = 0

plt.plot(my_data[:,1])
plt.ax
plt.show()

print(my_data[0][1])