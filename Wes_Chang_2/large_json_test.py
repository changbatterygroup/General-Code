import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import math
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

file_dir = "/Users/wesleychang/OneDrive - Drexel University/Chang Lab/General/Group/Data/Ultrasound/EFC project/"
# files = ['20240204_WC_JE_0p25M_Na2SO4', '20240204_WC_JE_0p5M_Na2SO4', '20240204_WC_JE_1M_Na2SO4', '20240204_WC_JE_2M_Na2SO4']
files = ['20240202_WC_JE_EFC2']
# ,'20240202_WC_JE_EFC2']
cyc_file_path = "/Users/wesleychang/OneDrive - Drexel University/Chang Lab/General/Group/Data/Ultrasound/EFC project/ultrasound_static_efc_test20.idf"

def plot_json_data(json_file, color = 'k', label = ''):
    with open(json_file, 'r') as file:
        lines = file.readlines()

    total_amp = []
    times = []
    all_waves = []
    # Process each JSON object in the file
    for line in lines:
        try:
            data = json.loads(line)

            # Access the 'amps' and 'time' keys in the JSON data
            amps = data.get('amps', [])
            all_waves.append(amps[0])
            times.append(data.get('time', []))
            # for n,d in enumerate(amps):
            #     total_amp.append(np.dot(d,d)) 
                
            # for t in list(time):
            #     times.append(time)
            # # Process or print the data as needed
            # print("Amps Data:", amps)
            # print("Time Data:", time_data)
        except: pass
        # except json.JSONDecodeError as e:
        #     print(f"Error decoding JSON: {e}")
    return all_waves, times

# This function utilizes a short term avg long term avg (STA/LTA) detection algorithm to determine wave transmit time
def classic_sta_lta(wave, nsta = None, nlta = None):  
    df_wave = pd.DataFrame(wave)
    lta = df_wave.rolling(window=nlta).mean()
    sta = df_wave.rolling(window=nsta).mean()
    tiny = np.finfo(0.0).tiny
    lta[lta < tiny] = tiny
    return (sta / lta, sta, lta)
    
def first_break_picker(waves,t_0,t_1,lta2sta):
    index = []
    stalta_ = []
    points = len(waves[0])
    t_sta = 0.5E-6 
    sampling_rate = 1/abs((1./points)*(t_1-t_0)*10E-6) #spacing of data collection for one wave
    nsta = int(t_sta * sampling_rate)
    nlta = lta2sta * nsta
    for wave in waves:
        stalta,__,__ = classic_sta_lta(wave**2,nsta=nsta,nlta=nlta)
        thres = 0.75 * np.nanmax(stalta) #0.75 arbitrarily set
 
        stalta_.append(stalta)
        stalta = stalta.values #convert to numpy array for argmax finder
        
        '''Threshold'''
        max_index = (stalta > thres).argmax() if (stalta > thres).any() else -1
        # max_index = stalta.idxmax()
        index.append((float(max_index)/points)*(t_1 - t_0) + t_0)
        
    return stalta_,index

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
# ax2 = ax1.twinx()
color_map = plt.cm.get_cmap('coolwarm')

for i,f in enumerate(files):
    waves,times = plot_json_data(file_dir + f + '.json')

    # delay = 14.
    # duration = 2. 
    # tofs = np.linspace(duration, duration + delay, len(waves))
    # lta2sta = 12
    # ref = 2

    # stalta_,index = first_break_picker(np.array(waves),14.,16.,lta2sta)
    # print(f,index)

    for n,d in enumerate(waves[500:2500]):
        normalize = Normalize(vmin=0, vmax=len(waves[500:2500]))
        scalar_map = ScalarMappable(norm=normalize, cmap=color_map)
        amp = []
        ax1.plot(d, color = scalar_map.to_rgba(n))
        # amp.append(np.dot(d,d)) 
    # ax1.plot(index, color = scalar_map.to_rgba(i), label = f)

# with open(cyc_file_path, 'r') as file:
#     cycle_data = pd.read_csv(file, delim_whitespace=True)
#     cycle_data.columns = ['t','v','i']
#     ax2.plot((cycle_data['t']-cycle_data['t'].iloc[0])/3600. + 1.3,cycle_data['v'], color='#E0001B')
# ax2.set_ylim(0,2)


    
# ax1.plot(((np.array(times))-(np.array(times))[0])/3600., np.array(amp)/amp[0], 'k', lw = 0.8)
# ax1.set_xlim(0.1,20)
# ax1.set_xlabel('Time (sec)')
# ax1.set_ylabel('Wave transmit time ($\mathregular{\mu}$s)')
# ax1.legend(loc='best')
# ax1.set_ylabel('Total Amplitude (A/A$\mathregular{_0}$)')
# ax2.set_ylabel('Voltage (V)', color='#E0001B')


plt.show()