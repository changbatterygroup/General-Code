import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from io import StringIO

file_path = "/Users/wesleychang/OneDrive - Drexel University/Chang Lab/General/Group/Data/Landt_cyclers/CoinCell_006_FT_012324004_002_4.csv"

data = pd.read_csv(file_path, error_bad_lines=False, sep=',',header=16)
data = data[pd.to_numeric(data['Record'], errors='coerce').notna()]
data = data[0:7000]

# Convert 'TestTime' to datetime objects
# data['TestTime'] = pd.to_datetime(data['TestTime'], format='%H:%M:%S.%f')
print(data.head())
print(len(data))
plt.plot(data['Voltage/V'])
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage (V)')
plt.title('Time vs Voltage')
plt.show()
