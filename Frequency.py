from scipy.fft import rfft, irfft, rfftfreq
import pandas as pd
import numpy as np
import pylab as plt
# Import matplotlib

#demand_range = range(5,30,5)
demand_range = range(5,10,5)
sample_spacing = 1 # [1/Hour]

cutoff_frequencies = {
                    "intraday": [0.5,1/23],
                    "overnight": [1/23,1/25],
                    "monthly": [1/25,1/(24*30)],
                    "seasonal": [1/(24*30),1/(24*30*12)],
                    "longterm": [1/(24*30*12),1/(24*30*12*10)]
                    }

taxonomy = list(cutoff_frequencies.keys())

for demand in demand_range:
    lpgm_df = pd.read_csv(f"Results/LPGM_SEAsiaMY{demand}.csv")

    time = np.linspace(0,8760*10,8760*10)
    storage_timeseries_np = lpgm_df["Pumped hydro energy storage"].to_numpy()

    W = rfftfreq(storage_timeseries_np.size,d=time[1] - time[0])
    storage_frequency = rfft(storage_timeseries_np)
    
    category_storage_frequencies = {}
    category_storage_timeseries = {}
    first = True
    category_storage_frequencies_df = pd.DataFrame()
    category_storage_timeseries_df = pd.DataFrame()

    """ category_storage_frequencies_df["Original"] = storage_frequency
    category_storage_timeseries_df["Original"] = storage_timeseries_np """

    for category in taxonomy:
        cut_storage_frequency = storage_frequency.copy()
        cutoff_low = cutoff_frequencies[category][1]
        cutoff_high = cutoff_frequencies[category][0]
        print(cutoff_low,cutoff_high)

        cut_storage_frequency[W>cutoff_high] = 0
        cut_storage_frequency[W<=cutoff_low] = 0
        cut_storage_frequency[-1] = 0

        category_storage_frequencies[category] = cut_storage_frequency
        category_storage_timeseries[category] = irfft(cut_storage_frequency)

        category_storage_frequencies_df[category] = cut_storage_frequency
        category_storage_timeseries_df[category] = category_storage_timeseries[category]

        """ if first:
            category_storage_frequencies_df = pd.DataFrame(cut_storage_frequency)
            category_storage_timeseries_df = pd.DataFrame(category_storage_timeseries[category])
            first=False
        else:
            new_f_df = pd.DataFrame(cut_storage_frequency)
            new_t_df = pd.DataFrame(category_storage_timeseries[category])
            category_storage_frequencies_df = pd.concat([category_storage_frequencies_df,new_f_df])
            category_storage_timeseries_df = pd.concat([category_storage_timeseries_df,new_t_df]) """

    
    print(category_storage_frequencies_df)
    print(category_storage_timeseries_df)

    category_storage_frequencies_df.to_csv(f"frequencies_{demand}.csv")
    category_storage_timeseries_df.to_csv(f"energy_profile_{demand}.csv")

    """ plt.subplot(2,6,2)
    plt.plot(time,storage_timeseries_np)
    plt.xlim(0,1000)
    plt.subplot(2,6,1)
    plt.plot(W,storage_frequency)
    plt.xlim(0,0.5)
    plt.subplot(2,6,3)
    plt.plot(W,category_storage_frequencies["intraday"])
    plt.xlim(0,cutoff_frequencies["intraday"][0])
    plt.subplot(2,6,4)
    plt.plot(time,category_storage_timeseries["intraday"])
    plt.xlim(0,240)
    plt.subplot(2,6,5)
    plt.plot(W,category_storage_frequencies["overnight"])
    plt.xlim(0,cutoff_frequencies["overnight"][0])
    plt.subplot(2,6,6)
    plt.plot(time,category_storage_timeseries["overnight"])
    plt.xlim(0,1000)
    plt.subplot(2,6,7)
    plt.plot(W,category_storage_frequencies["monthly"])
    plt.xlim(0,cutoff_frequencies["monthly"][0])
    plt.subplot(2,6,8)
    plt.plot(time,category_storage_timeseries["monthly"])
    plt.xlim(0,2000)
    plt.subplot(2,6,9)
    plt.plot(W,category_storage_frequencies["seasonal"])
    plt.xlim(0,cutoff_frequencies["seasonal"][0])
    plt.subplot(2,6,10)
    plt.plot(time,category_storage_timeseries["seasonal"])
    plt.xlim(0,8760)
    plt.subplot(2,6,11)
    plt.plot(W,category_storage_frequencies["longterm"])
    plt.xlim(0,cutoff_frequencies["longterm"][0])
    plt.subplot(2,6,12)
    plt.plot(time,category_storage_timeseries["longterm"])
    plt.xlim(0,87600)
    plt.show() """


    
