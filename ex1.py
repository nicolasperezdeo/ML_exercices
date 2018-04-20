import numpy as np
import matplotlib.pyplot as plt

from numpy import genfromtxt
my_data = genfromtxt('traffic_per_hour.csv', delimiter='\t')

my_data[np.isnan(my_data)] = np.mean(my_data)

#print(my_data[:,0])
plt.scatter(my_data[:,0],my_data[:,1])
plt.xlabel('Time(s)')
plt.ylabel('F(t)')
plt.show()

z = np.polyfit(my_data[:,0], my_data[:,1], 3)
