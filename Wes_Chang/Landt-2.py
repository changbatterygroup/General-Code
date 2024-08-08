import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from io import StringIO
import csv
import glob
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

file_path = glob.glob("/Users/wesleychang/OneDrive - Drexel University/Chang Lab/General/Group/Data/Landt_cyclers/CoinCell_017_FT_*.csv")
color_map = plt.cm.get_cmap('coolwarm')

def parse_csv(file_path):
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        # Skip initial headers
        header = next(csv_reader)
        # print(header)

        if header[0] != "Step" or header[1] != "Mode":
            raise ValueError("Invalid header format")
 
        # Process data rows
        voltage = []
        capacity = []
        current = []
        for row in csv_reader:

            # Assuming the voltage column is in the last position (index -1)
            try:
                volt = float(row[-1])
                cap = float(row[-2])
                curr = float(row[-3])
            except: 
                print(row)
                continue 
            voltage.append(volt)
            capacity.append(cap)
            current.append(curr)
        # print(voltage_data)
        return voltage,capacity,current

normalize = Normalize(vmin=0, vmax=len(file_path))
scalar_map = ScalarMappable(norm=normalize, cmap=color_map)

for i,f in enumerate(file_path):
    voltage, capacity, current = parse_csv(f)
    plt.scatter(capacity, voltage, color = scalar_map.to_rgba(i),marker='o',s=4)
    plt.xlabel('Time')
    plt.ylabel('Voltage (V)')

  
plt.show()