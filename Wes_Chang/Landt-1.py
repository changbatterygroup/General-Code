import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from io import StringIO
import csv

file_path = "/Users/wesleychang/OneDrive - Drexel University/Chang Lab/General/Group/Data/Landt_cyclers/CoinCell_006_FT_012324006_002_6~2.csv"

def time_string_to_seconds(time_string):
    # Split the time string into hours, minutes, seconds, and milliseconds
    parts = time_string.split(':')
    days = float(parts[0].split('-')[0] if '-' in time_string else 0) 
    hours = float(parts[0].split('-')[1] if '-' in time_string else parts[0])
    min = int(parts[1])
    sec = float(parts[2].split('.')[0])

    # Handle milliseconds separately
    milliseconds = float("0." + parts[2].split('.')[1]) 
    
    # Calculate the total number of seconds
    total_seconds = days * 24*3600 + hours * 3600 + min * 60 + sec + milliseconds 
    
    return int(total_seconds)


# data = pd.read_csv(file_path, on_bad_lines='warn', sep=',',header=2)
# data = data[pd.to_numeric(data['Record'], errors='coerce').notna()]

# Convert 'TestTime' to datetime objects
# data['TestTime'] = pd.to_datetime(data['TestTime'], format='%H:%M:%S.%f', errors = 'coerce')
# for i,t in enumerate(data['TestTime']):
#     try: 
#         data['TestTime'][i] = time_string_to_seconds(t)
#     except (ValueError, IndexError): pass 

cyc_data = pd.DataFrame()

with open(file_path) as file:
    data = pd.read_csv(file, on_bad_lines='warn', sep=',', header=2)

    # data['TestTime'] = time_string_to_seconds(str(data['TestTime']))

#     for index,row in data.iterrows():
#         if len(data.columns) > 3:
#             try:
#                 row['TestTime'] = time_string_to_seconds(str(row['TestTime']))
#                 row['TestTime'] = float(row['TestTime'])
#                 row['Voltage/V'] = float(row['Voltage/V'])
#                 cyc_data = cyc_data.append(row, ignore_index=True)
#             except (ValueError, IndexError): pass
#         else: 
#             continue

plt.plot(data[data['Current/uA']>0]['Capacity/uAh'],data[data['Current/uA']>0]['Voltage/V'])
# print(cyc_data[-400:], len(cyc_data))
# print(data.head())
# print(len(data))
# print(data.keys())
# print(data['TestTime'])
# plt.plot(cyc_data['TestTime'], cyc_data['Voltage/V'])
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage (V)')
plt.title('Time vs Voltage')
plt.show()
