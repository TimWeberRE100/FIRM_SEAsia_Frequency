from scipy.fft import rfft, irfft, rfftfreq
import pandas as pd
import numpy as np
import pylab as plt

time = np.linspace(1,8760,8760)

timeseries_df = pd.read_csv("/home/tim/Documents/Projects/Frequency Analysis/TestFunctions.csv")

timeseries_names = list(timeseries_df.columns.values)

category_storage_frequencies_df = pd.DataFrame()
category_storage_timeseries_df = pd.DataFrame()

for category in timeseries_names:
    category_storage_frequencies_df[category] = rfft(np.array(timeseries_df[category]))
    category_storage_timeseries_df[category] = irfft(np.array(category_storage_frequencies_df[category]))
    W = rfftfreq(np.array(timeseries_df[category]).size,d=time[1] - time[0])

    plt.subplot(2,2,1)
    plt.plot(time,timeseries_df[category])
    plt.xlim(3000,5000)
    plt.subplot(2,2,2)
    plt.plot(W,category_storage_frequencies_df[category])
    plt.xlim(0,1/23)
    plt.ylim(0,200000)
    plt.subplot(2,2,3)
    plt.plot(time,timeseries_df[category])
    plt.xlim(0,8760)
    plt.show() 