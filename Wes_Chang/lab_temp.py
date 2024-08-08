import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt 
from datetime import datetime

file = "/Users/wesleychang/OneDrive - Drexel University/Chang Lab/General/Group/Data/TemperatureData/LabTemperatureData.txt"

fig1 = plt.figure(dpi=300, figsize = (6,4))
colors = plt.get_cmap('coolwarm')

ax1 = fig1.add_subplot(111)

temp_data = pd.read_csv(file,header=1, delimiter=' ')

temp = temp_data.iloc[:,4]
date = temp_data.iloc[:,0] + ' ' + temp_data.iloc[:,1]
# print(date)
timestamp = pd.to_datetime(date, format="%Y-%m-%d %H-%M-%S:")
# print(timestamp)
data = [float(temp.replace('°C', '')) for temp in temp]

ax1.plot(timestamp, data, lw = 0.5)
plt.xticks(rotation=45)
ax1.set_xlabel('Time')
ax1.set_ylabel('Lab Temp ($^\circ$C)')
plt.show()