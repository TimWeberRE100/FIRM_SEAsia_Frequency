from scipy.fft import rfft, irfft, rfftfreq
import pandas as pd
import numpy as np
import pylab as plt

""" def find_maximum_intervals(storage_timeseries_np):
    maximum = max(storage_timeseries_np)
    indices = []
    for i in range(0,len(storage_timeseries_np)):
        if abs(storage_timeseries_np[i]-maximum) < 0.0001:
            indices.append(i)

    return indices """

#demand_range = range(5,26)
demand_range = range(5,26)
sample_spacing = 1 # [1/Hour]
#threshold_mag = 0.5

cutoff_frequencies = {
                    "intraday": [0.5,1/20],
                    "overnight": [1/20,1/28],
                    "monthly": [1/28,1/(24*30)],
                    "seasonal": [1/(24*30),1/(24*366)],
                    "longterm": [1/(24*366),-1/(24*30*12*10)]
                    }

taxonomy = ["intraday","overnight","monthly","seasonal","longterm"]

energy_capacity_results = pd.DataFrame()
energy_capacity_results["Demand [MWh/capita/year]"] = list(demand_range)

for demand in demand_range:
    lpgm_df = pd.read_csv(f"Results/LPGM_SEAsiaMY{demand}.csv")

    time = np.linspace(0,8760*10,8760*10)
    storage_timeseries_np = lpgm_df["PHES-Storage"].to_numpy()

    W = rfftfreq(storage_timeseries_np.size,d=time[1] - time[0])
    storage_frequency = rfft(storage_timeseries_np)
    storage_frequency_magnitudes_raw = abs(storage_frequency)
    storage_frequency_magnitudes = storage_frequency_magnitudes_raw.copy()
    storage_frequency_magnitudes[0] = 0 # drop the dc offset magnitude
    storage_frequency_magnitudes = storage_frequency_magnitudes / max(storage_frequency_magnitudes) # normalise results
    
    possible_noise = storage_frequency_magnitudes.copy()
    #possible_noise[possible_noise > 0.2] = 0

    # Sliding noise window
    noise_means = np.zeros(len(storage_frequency_magnitudes))
    stddevs = np.zeros(len(storage_frequency_magnitudes))
    for i in range(0,len(noise_means)):
        if (i >= 20) and (i<len(noise_means)-21):            
            noise_means[i] = np.mean(possible_noise[i-20:i+20])
            stddevs[i] = np.std(possible_noise[i-20:i+20])

            if i < 25:
                stddevs[i] *= 2
            else:
                stddevs[i] *= 4

    for i in range(0,20):
        noise_means[i] = noise_means[20]
        noise_means[len(noise_means) - 1 - i] = noise_means[len(noise_means) - 22]

    storage_frequency_thresholds = storage_frequency_magnitudes.copy()
    storage_frequency_thresholds[(storage_frequency_thresholds < 4*noise_means)] = 0

    storage_frequency_cleaned = storage_frequency.copy()
    storage_frequency_cleaned[(storage_frequency_thresholds == 0) | (W <= cutoff_frequencies["longterm"][0])] = 0

    storage_frequency_reserves = storage_frequency.copy()
    storage_frequency_reserves[(storage_frequency_thresholds > 0) & (W > cutoff_frequencies["longterm"][0])] = 0
    
    category_storage_frequencies_df = pd.DataFrame()
    category_storage_timeseries_df = pd.DataFrame()

    minimums = {}

    for category in taxonomy:
        if demand == demand_range[0]:
            energy_capacity_results[category] = np.zeros(len(list(demand_range)))
            print(category, len(list(demand_range)))

        if category != "longterm":
            cut_storage_frequency = storage_frequency_cleaned.copy()

            cutoff_low = cutoff_frequencies[category][1]
            cutoff_high = cutoff_frequencies[category][0]

            cut_storage_frequency[W>cutoff_high] = 0
            cut_storage_frequency[W<=cutoff_low] = 0
        else:
            cut_storage_frequency = storage_frequency_reserves.copy()
       

        category_storage_frequencies_df[category] = cut_storage_frequency
        category_storage_timeseries_df[category] = irfft(cut_storage_frequency)

        minimums[category] = min(category_storage_timeseries_df[category])

    category_energy_capacities = {}
    for category in taxonomy:
        if category != "longterm":
            category_storage_timeseries_df[category] = category_storage_timeseries_df[category] - minimums[category]
            category_storage_timeseries_df["longterm"] = category_storage_timeseries_df["longterm"] + minimums[category]
            category_energy_capacities[category] = max(category_storage_timeseries_df[category])

    category_energy_capacities["longterm"] = max(storage_timeseries_np) - sum([category_energy_capacities[x] for x in taxonomy if x != "longterm"])
    print(category_energy_capacities["longterm"])
    assert category_energy_capacities["longterm"] >= 0

    # Adjust the category time series to prevent them exceeding energy capacity
    test=1
    for i in range(0,len(time)):
        carryover = 0
        if i == test:
            print(category_storage_timeseries_df.iloc[i])
        adjusted_value = max(min(category_energy_capacities["intraday"], category_storage_timeseries_df.at[i,"intraday"]),0)
        carryover = category_storage_timeseries_df.at[i,"intraday"] - adjusted_value
        category_storage_timeseries_df.at[i,"intraday"] = adjusted_value

        category_storage_timeseries_df.at[i,"overnight"] += carryover
        adjusted_value = max(min(category_energy_capacities["overnight"], category_storage_timeseries_df.at[i,"overnight"]),0)
        carryover = category_storage_timeseries_df.at[i,"overnight"] - adjusted_value
        category_storage_timeseries_df.at[i,"overnight"] = adjusted_value

        category_storage_timeseries_df.at[i,"monthly"] += carryover
        adjusted_value = max(min(category_energy_capacities["monthly"], category_storage_timeseries_df.at[i,"monthly"]),0)
        carryover = category_storage_timeseries_df.at[i,"monthly"] - adjusted_value
        category_storage_timeseries_df.at[i,"monthly"] = adjusted_value

        category_storage_timeseries_df.at[i,"seasonal"] += carryover
        adjusted_value = max(min(category_energy_capacities["seasonal"], category_storage_timeseries_df.at[i,"seasonal"]),0)
        carryover = category_storage_timeseries_df.at[i,"seasonal"] - adjusted_value
        category_storage_timeseries_df.at[i,"seasonal"] = adjusted_value

        category_storage_timeseries_df.at[i,"longterm"] += carryover
        adjusted_value = max(min(category_energy_capacities["longterm"], category_storage_timeseries_df.at[i,"longterm"]),0)
        carryover = category_storage_timeseries_df.at[i,"longterm"] - adjusted_value
        category_storage_timeseries_df.at[i,"longterm"] = adjusted_value

        category_storage_timeseries_df.at[i,"seasonal"] += carryover
        adjusted_value = max(min(category_energy_capacities["seasonal"], category_storage_timeseries_df.at[i,"seasonal"]),0)
        carryover = category_storage_timeseries_df.at[i,"seasonal"] - adjusted_value
        category_storage_timeseries_df.at[i,"seasonal"] = adjusted_value

        category_storage_timeseries_df.at[i,"monthly"] += carryover
        adjusted_value = max(min(category_energy_capacities["monthly"], category_storage_timeseries_df.at[i,"monthly"]),0)
        carryover = category_storage_timeseries_df.at[i,"monthly"] - adjusted_value
        category_storage_timeseries_df.at[i,"monthly"] = adjusted_value

        category_storage_timeseries_df.at[i,"overnight"] += carryover
        adjusted_value = max(min(category_energy_capacities["overnight"], category_storage_timeseries_df.at[i,"overnight"]),0)
        carryover = category_storage_timeseries_df.at[i,"overnight"] - adjusted_value
        category_storage_timeseries_df.at[i,"overnight"] = adjusted_value

        category_storage_timeseries_df.at[i,"intraday"] += carryover
        adjusted_value = max(min(category_energy_capacities["intraday"], category_storage_timeseries_df.at[i,"intraday"]),0)
        carryover = category_storage_timeseries_df.at[i,"intraday"] - adjusted_value
        category_storage_timeseries_df.at[i,"intraday"] = adjusted_value

    category_storage_frequencies_df["Original"] = storage_frequency
    category_storage_timeseries_df["Original"] = storage_timeseries_np
    category_storage_frequencies_df["Frequencies"] = W

    category_magnitudes_df = pd.DataFrame(storage_frequency_magnitudes_raw)
    category_magnitudes_df["normalised"] = storage_frequency_magnitudes
    category_magnitudes_df["cleaned"] = storage_frequency_cleaned
    category_magnitudes_df["threshold"] = storage_frequency_thresholds
    category_magnitudes_df.to_csv(f"Results/magnitudes_{demand}.csv")

    category_storage_frequencies_df.to_csv(f"Results/frequencies_{demand}.csv")
    category_storage_timeseries_df.to_csv(f"Results/energy_profile_{demand}.csv")

    for category in taxonomy:
        energy_capacity_results.at[demand-5,category] = max(np.array(category_storage_timeseries_df[category]))/1000 # GWh

energy_capacity_results.to_csv("Results/energy_capacity_results.csv")


    
