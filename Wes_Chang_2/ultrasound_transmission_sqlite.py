import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.signal import stft
from matplotlib.cm import ScalarMappable
from matplotlib.pyplot import get_cmap
from matplotlib.colors import Normalize

file_dir = "/Users/Michael/OneDrive - Drexel University/Documents - Chang Lab/General/Group/Data/Ultrasound/Layered electrode study/"
# files = ['20240204_WC_JE_0p25M_Na2SO4', '20240204_WC_JE_0p5M_Na2SO4', '20240204_WC_JE_1M_Na2SO4', '20240204_WC_JE_2M_Na2SO4']
# files = ['20240206_WC_JE_EFC_1M_KCl_delay=11_dur=3_test2.sqlite3']
# files = ['AT_WC_LMFP_SiOx_0P1C.sqlite3']
files = ['20240216_WC_GJ_water-ref_dur15_del1p5_50MHz.sqlite3', '20240216_WC_GJ_Cu-n=1_dur15_del1p5_50MHz.sqlite3',
         '20240216_WC_GJ_Cu-n=2_dur15_del1p5_50MHz.sqlite3', '20240216_WC_GJ_Cu-n=3_dur15_del1p5_50MHz.sqlite3',
         '20240216_WC_GJ_Cu-n=4_dur15_del1p5_50MHz.sqlite3', '20240216_WC_GJ_Cu-n=5_dur15_del1p5_50MHz.sqlite3']
color_map = plt.get_cmap('coolwarm')


# colors = ['k','#E0001B','blue', ora]

# This function utilizes a short term avg long term avg (STA/LTA) detection algorithm to determine wave transmit time
def classic_sta_lta(wave, nsta=None, nlta=None):
    df_wave = pd.DataFrame(wave)
    lta = df_wave.rolling(window=nlta).mean()
    sta = df_wave.rolling(window=nsta).mean()
    tiny = np.finfo(0.0).tiny
    lta[lta < tiny] = tiny
    return (sta / lta, sta, lta)


# This function determines the wave transmission time by finding the time at which the initial part of the wave first arrives
def first_break_picker(waves, t_0, t_1, lta2sta):
    index = []
    stalta_ = []
    points = len(waves[0])
    t_sta = 0.5E-6
    sampling_rate = 1 / abs((1. / points) * (t_1 - t_0) * 10E-6)  #spacing of data collection for one wave
    nsta = int(t_sta * sampling_rate)
    nlta = lta2sta * nsta
    for wave in waves:
        stalta, __, __ = classic_sta_lta(wave ** 2, nsta=nsta, nlta=nlta)
        thres = 0.75 * np.nanmax(stalta)  #0.75 arbitrarily set

        stalta_.append(stalta)
        stalta = stalta.values  #convert to numpy array for argmax finder

        '''Threshold'''
        max_index = (stalta > thres).argmax() if (stalta > thres).any() else -1
        # max_index = stalta.idxmax()
        index.append((float(max_index) / points) * (t_1 - t_0) + t_0)

    return stalta_, index


# This function incorporates a cross-correlation (convolution integral) to find the wave time-of-flight shift (
# difference in transmission times between two waves)
def cross_correlate_tolerance(crossA, crossB, time, t_unit, tolerance, ToF_Prev=0.):
    "outputs time lag of crossing A&B over t_range time with shifts calculated within a given tolerance"

    time_span = time[-1] - time[0]
    no_steps = len(time) - 1  #no. of data p. -1
    step_time = time_span / no_steps  #time betw. data points

    # Determine \DeltaT
    cross_t = np.correlate(crossA, crossB, mode='full')
    cross_t = cross_t.tolist()

    #find the index of the previous max correlation point
    old_index = (ToF_Prev + time_span) / step_time
    range_low = int(old_index - int(0.5 * tolerance / time_span * len(cross_t)))
    range_high = int(old_index + int(0.5 * tolerance / time_span * len(cross_t)))
    if len(cross_t[range_low:range_high]) > 0:
        max_index = cross_t.index(max(cross_t[range_low:range_high]))
    else:
        max_index = old_index

    DeltaT = step_time * max_index  #shift from max lag
    deltaT = DeltaT - time_span  #shift from 0 lag

    if (abs(deltaT - ToF_Prev)) > tolerance: deltaT = ToF_Prev
    print(DeltaT, deltaT)
    return deltaT


# This function calculates the time-of-flight shift based on the cross-correlation 
def TOFshift(tofs, ref, wave, points=6000, ikind='cubic', tol=3., fill=128):
    #interpolate the 2 given waves as functions
    tnew = np.linspace(tofs[0], tofs[-1], points)
    print(tofs)
    wref = interp1d(tofs, ref, kind=ikind, bounds_error=False, fill_value=fill)
    wf1 = interp1d(tofs, wave, kind=ikind, bounds_error=False, fill_value=fill)
    #minimize the error of subtracting shifted wave from reference
    tshift = cross_correlate_tolerance(wref(tnew), wf1(tnew), tnew, 'us', tol)
    return tshift


'''''''''''''''''''''''Frequency domain analysis'''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


# This function does an FFT of the acoustic wave
def fft_acoustics(waves, interval=1 / 5e8, pad_x=2):
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


# This function does some upsampling to improve wave resolution
def upsample(wave, upsample, x=None):
    '''
    returns a wave upsampled via a factor given in upsample. uses cubic spline interpolation
    '''
    if x is None:
        x = np.arange(len(wave))

    spline = CubicSpline(x, wave)
    new_x = np.arange(x[0], x[-1], (x[-1] - x[0]) / (len(wave) * upsample))
    return new_x, spline(new_x)


def temporal_freq(wave, f_low=2.e6, f_mid=5.e6, f_high=8e6, fs=500e6, usample=500):
    '''
    gives time-amplitude slices in a windowed fourier transform 
    f_low is the low frequency tof diagram
    f_high is the high frequency
    usample applies a spline interpolation to improve temporal resolution for peak finding
    '''
    nfft = 1e3
    window = 256 / 2
    overlap = window - 5
    freqs, t, s = stft(wave, fs=fs, return_onesided=True, nfft=nfft, nperseg=window, noverlap=overlap)
    floc = (np.where(freqs < f_low)[0][-1])
    freq_slice = np.abs(s[floc, :])
    # _, lowfreq = upsample(freq_slice, usample)

    return freq_slice


figure = plt.figure(figsize=(10, 6))
ax1 = figure.add_subplot(111)
# ax2 = figure.add_subplot(312)
# ax3 = figure.add_subplot(313)
labels = ['water reference', 'Cu n=1', 'Cu n=2', 'Cu n=3', 'Cu n=4', 'Cu n=5']
for n, f in enumerate(files):
    connection = sqlite3.connect(file_dir + f)
    cursor = connection.cursor()
    query = """SELECT name FROM sqlite_master WHERE type='table'"""
    cursor.execute(query)
    table = cursor.fetchall()

    # select every nth wave to speed up processing 
    query = f'SELECT * FROM "{table[0][0]}" WHERE "time" % 2 == 0'
    df = pd.read_sql(con=connection, sql=query)
    connection.close()

    waves_formatted: List[str] = df['amps'].str.strip('[]').str.split(',')
    waves = np.zeros(
        (len(waves_formatted), len(waves_formatted[0])),
        dtype=np.float16
    )

    for i, wave in enumerate(waves_formatted):
        waves[i, :] = wave

    amps = []
    times = (df['time'] - df['time'].iloc[0]) / 3600.
    tofshift = []
    tofs = np.linspace(11, 11 + 3, len(waves[0]))

    freqs, amps = fft_acoustics(waves)
    normalize = Normalize(vmin=0, vmax=len(files))
    scalar_map = ScalarMappable(norm=normalize, cmap=color_map)
    print(n, f)
    # ax1.plot(freqs,amps[0], color = scalar_map.to_rgba(n), label=labels[n])
    # ax1.set_xlim(0,100E6)
    for c, wave in enumerate(waves):
        #     print(len(waves[0]), len(wave), (wave))
        ax1.plot(wave - 0.02 * n, color=scalar_map.to_rgba(n), label=labels[n])
        break
    #     break
    #     amps.append(np.dot(wave, wave)) #dot product of the waveform to produce an artifically exaggerated intensity value 
    # dtof = -TOFshift(tofs,waves[0],(wave))
    # tofshift.append(dtof)

    # ax1.plot(times, amps/amps[0]) # plots the normalized amplitude intensity relative to the first waveform collected 

    # ax1.plot(times, tofshift) 
    # ax3.pcolormesh(times, tofs, np.array(waves).T, cmap=color_map)
    # stalta,index = first_break_picker(waves, 11, 14, lta2sta = 14)

    # for s in stalta:
    #     ax1.plot(tofs,s)

# ax1.set_xlabel('Time ()')
# ax1.set_ylabel('Amplitude')
# ax2.set_ylabel('Time of flight shift ($\mathregular{\mu}$s)')
# ax1.set_xlabel('Frequency (Hz)')
# ax1.set_ylabel('Amplitude')
ax1.legend()
# for ax in [ax1, ax2]:
#     ax.set_xlim(0, 20)
# ax2.set_ylim(-0.05,0.05)
# plt.savefig('20240216_electrode.tiff')
plt.show()
