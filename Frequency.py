from scipy.fft import rfft, irfft, rfftfreq
import pandas as pd
import numpy as np
import pylab as plt
# Import matplotlib

#demand_range = range(5,30,5)
demand_range = range(5,30,5)
sample_spacing = 1 # [1/Hour]

cutoff_frequencies = {
                    "intraday": [0.5,1/20],
                    "overnight": [1/20,1/28],
                    "monthly": [1/28,1/(24*30)],
                    "seasonal": [1/(24*30),1/(24*30*12)],
                    "longterm": [1/(24*30*12),-1/(24*30*12*10)]
                    }

taxonomy = list(cutoff_frequencies.keys())

for demand in demand_range:
    lpgm_df = pd.read_csv(f"Results/LPGM_SEAsiaMY{demand}.csv")

    time = np.linspace(0,8760*10,8760*10)
    storage_timeseries_np = lpgm_df["PHES-Storage"].to_numpy()

    W = rfftfreq(storage_timeseries_np.size,d=time[1] - time[0])
    storage_frequency = rfft(storage_timeseries_np)
    print(storage_frequency)
    
    category_storage_frequencies_df = pd.DataFrame()
    category_storage_timeseries_df = pd.DataFrame()

    for category in taxonomy:
        cut_storage_frequency = storage_frequency.copy()
        cutoff_low = cutoff_frequencies[category][1]
        cutoff_high = cutoff_frequencies[category][0]
        print(cutoff_low,cutoff_high)

        cut_storage_frequency[W>cutoff_high] = 0
        cut_storage_frequency[W<=cutoff_low] = 0

        category_storage_frequencies_df[category] = cut_storage_frequency
        category_storage_timeseries_df[category] = irfft(cut_storage_frequency)

    minimums = {}
    maximums = {}
    for category in taxonomy:
        print(category)
        minimums[category] = np.abs(min(np.array(category_storage_timeseries_df[category])))
        for i in range(0,len(time)):        
            if category != "longterm":                
                category_storage_timeseries_df.at[i,category]=category_storage_timeseries_df.at[i,category] + minimums[category]
            else:
                category_storage_timeseries_df.at[i,category]=category_storage_timeseries_df.at[i,category] - minimums["intraday"] - minimums["overnight"] - minimums["monthly"] - minimums["seasonal"]
        maximums[category] = max(np.array(category_storage_timeseries_df[category]))
    
    minimums["longterm"] = np.abs(min(np.array(category_storage_timeseries_df["longterm"])))
    for i in range(0,len(time)):
        if category_storage_timeseries_df.at[i,"longterm"] < 0:
            apportioning = abs(category_storage_timeseries_df.at[i,"longterm"])
            apportioning_original = apportioning

            original = category_storage_timeseries_df.at[i,"seasonal"]
            category_storage_timeseries_df.at[i,"seasonal"] = max(0,category_storage_timeseries_df.at[i,"seasonal"]-apportioning)
            apportioning += (category_storage_timeseries_df.at[i,"seasonal"] - original)

            original = category_storage_timeseries_df.at[i,"monthly"]
            category_storage_timeseries_df.at[i,"monthly"] = max(0,category_storage_timeseries_df.at[i,"monthly"]-apportioning)
            apportioning += (category_storage_timeseries_df.at[i,"monthly"] - original)

            original = category_storage_timeseries_df.at[i,"overnight"]
            category_storage_timeseries_df.at[i,"overnight"] = max(0,category_storage_timeseries_df.at[i,"overnight"]-apportioning)
            apportioning += (category_storage_timeseries_df.at[i,"overnight"] - original)

            original = category_storage_timeseries_df.at[i,"intraday"]
            category_storage_timeseries_df.at[i,"intraday"] = max(0,category_storage_timeseries_df.at[i,"intraday"]-apportioning)
            apportioning += (category_storage_timeseries_df.at[i,"intraday"] - original)
            

            if apportioning > 0.01:
                print(i,apportioning,apportioning_original)

            category_storage_timeseries_df.at[i,"longterm"] = 0

    category_storage_frequencies_df["Original"] = storage_frequency
    category_storage_timeseries_df["Original"] = storage_timeseries_np
    category_storage_frequencies_df["Frequencies"] = W

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


    
