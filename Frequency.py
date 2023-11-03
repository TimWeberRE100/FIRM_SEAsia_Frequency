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

energy_capacity_results = pd.DataFrame()
energy_capacity_results["Demand [MWh/capita/year]"] = list(demand_range)

for demand in demand_range:
    lpgm_df = pd.read_csv(f"Results/LPGM_SEAsiaMY{demand}.csv")

    time = np.linspace(0,8760*10,8760*10)
    storage_timeseries_np = lpgm_df["PHES-Storage"].to_numpy()

    W = rfftfreq(storage_timeseries_np.size,d=time[1] - time[0])
    storage_frequency = rfft(storage_timeseries_np)
    
    category_storage_frequencies_df = pd.DataFrame()
    category_storage_timeseries_df = pd.DataFrame()

    for category in taxonomy:
        if demand == demand_range[0]:
            energy_capacity_results[category] = np.zeros(len(list(demand_range)))
            print(category, len(list(demand_range)))

        cut_storage_frequency = storage_frequency.copy()
        cutoff_low = cutoff_frequencies[category][1]
        cutoff_high = cutoff_frequencies[category][0]

        cut_storage_frequency[W>cutoff_high] = 0
        cut_storage_frequency[W<=cutoff_low] = 0

        category_storage_frequencies_df[category] = cut_storage_frequency
        category_storage_timeseries_df[category] = irfft(cut_storage_frequency)

    for i in range(0,len(time)):
        if category_storage_timeseries_df.at[i,"intraday"] < 0:
            category_storage_timeseries_df.at[i,"overnight"] += category_storage_timeseries_df.at[i,"intraday"]
            category_storage_timeseries_df.at[i,"intraday"] = 0
        if category_storage_timeseries_df.at[i,"overnight"] < 0:
            category_storage_timeseries_df.at[i,"monthly"] += category_storage_timeseries_df.at[i,"overnight"]
            category_storage_timeseries_df.at[i,"overnight"] = 0
        if category_storage_timeseries_df.at[i,"monthly"] < 0:
            category_storage_timeseries_df.at[i,"seasonal"] += category_storage_timeseries_df.at[i,"monthly"]
            category_storage_timeseries_df.at[i,"monthly"] = 0
        if category_storage_timeseries_df.at[i,"seasonal"] < 0:
            category_storage_timeseries_df.at[i,"longterm"] += category_storage_timeseries_df.at[i,"seasonal"]
            category_storage_timeseries_df.at[i,"seasonal"] = 0

        if category_storage_timeseries_df.at[i,"longterm"] < 0:
            category_storage_timeseries_df.at[i,"seasonal"] += category_storage_timeseries_df.at[i,"longterm"]
            category_storage_timeseries_df.at[i,"longterm"] = 0
        if category_storage_timeseries_df.at[i,"seasonal"] < 0:
            category_storage_timeseries_df.at[i,"monthly"] += category_storage_timeseries_df.at[i,"seasonal"]
            category_storage_timeseries_df.at[i,"seasonal"] = 0
        if category_storage_timeseries_df.at[i,"monthly"] < 0:
            category_storage_timeseries_df.at[i,"overnight"] += category_storage_timeseries_df.at[i,"monthly"]
            category_storage_timeseries_df.at[i,"monthly"] = 0
        if category_storage_timeseries_df.at[i,"overnight"] < 0:
            category_storage_timeseries_df.at[i,"intraday"] += category_storage_timeseries_df.at[i,"overnight"]
            category_storage_timeseries_df.at[i,"overnight"] = 0
        assert(category_storage_timeseries_df.at[i,"intraday"] > -0.01)
    
    maximums = {}
    for category in taxonomy:
        maximums[category] = max(np.array(category_storage_timeseries_df[category]))
    

    while sum(maximums) > max(storage_timeseries_np):
        # Extract all rows from original timeseries that have maximum energy capacity
        # Find difference between the category capacities in each of those rows and the current maximum capacities
        # Choose the row with the smallest difference to define the maximum capacities
        category_storage_timeseries_df[category]


    energy_capacity_results.at[demand//5-1,category] = maximums[category]


    category_storage_frequencies_df["Original"] = storage_frequency
    category_storage_timeseries_df["Original"] = storage_timeseries_np
    category_storage_frequencies_df["Frequencies"] = W

    category_storage_frequencies_df.to_csv(f"Results/frequencies_{demand}_top.csv")
    category_storage_timeseries_df.to_csv(f"Results/energy_profile_{demand}_top.csv")

energy_capacity_results.to_csv("Results/energy_capacity_results_top.csv")


    
