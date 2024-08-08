import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import glob
from typing import List
from scipy.interpolate import interp1d 
from scipy.interpolate import CubicSpline 
from scipy.signal import stft
from matplotlib.colors import Normalize

file_dir = "/Users/wesleychang/OneDrive - Drexel University/Chang Lab/General/Group/Data/Ultrasound/SES AI sample/"
# file_dir = "/Users/wesleychang/OneDrive - Drexel University/Chang Lab/General/Individual/Sam Amsterdam/SES data/E001_2_Tr/"
file_path = file_dir+'E001.csv'

SAMPLINGNUM = 10000
TOF_MIN = 0
TOF_MAX = 200
fig = plt.figure()
fig.subplots_adjust(hspace=0.5)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

'''''''''''''''''''''''Frequency domain analysis'''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# This function does an FFT of the acoustic wave
def fft_acoustics(waves, interval=1/5e8, pad_x=2):
    l = len(waves[0]) * pad_x
    n_waves = len(waves)
    z = np.zeros((n_waves, l), dtype='uint8')

    pad_waves = np.concatenate((z, waves, z), axis=1)

    m = np.mean(pad_waves, axis=-1)
    detrend = np.subtract(pad_waves.T, m).T
    amps = np.abs(np.fft.rfft(detrend, axis=-1))
    amps = np.divide(amps.T, np.max(amps, axis=-1)).T
    freqs = np.fft.rfftfreq(len(detrend[0]), d=interval)
    return freqs, amps

file_contents = pd.read_csv(file_path, header = 9)
tofs = np.linspace(TOF_MIN,TOF_MAX,SAMPLINGNUM)

wave = []
wave.append(np.array(file_contents['Transmission Wave Data: ']))
ax1.plot(tofs,wave[0])

freqs, amps = fft_acoustics(wave)
ax2.plot(freqs,amps[0])
ax2.set_xlim(0,1E7)

ax1.set_ylabel('Amplitude')
ax2.set_ylabel('Amplitude')
ax1.set_xlabel('Time-of-flight ($\mu$s)')
ax2.set_xlabel('Frequency (MHz)')
plt.show()  



