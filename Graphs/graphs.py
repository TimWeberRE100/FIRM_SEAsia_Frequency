import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot1():
    energy_df = pd.read_csv("./EnergyProfiles10.csv")
    frequency_df = pd.read_csv("./Frequencies10.csv")

    original_np = energy_df["Original"].to_numpy()
    #original_np = original_np[:2000]
    time = np.linspace(1,2000,2000)

    taxonomy = ["longterm","seasonal","monthly","overnight","intraday"]
    colours = {
        "longterm": "tab:blue",
        "seasonal": "tab:orange",
        "monthly": "tab:green",
        "overnight": "tab:red",
        "intraday": "tab:purple"
    }

    frequency_np = frequency_df["Frequency [1/hour]"].to_numpy()
    magnitude_np = frequency_df["Magnitude"].to_numpy()
    magnitude_clean_np = frequency_df["Cleaned"].to_numpy()

    fig, ax = plt.subplots(4, 1, figsize=(10, 8))

    ax[0].set_xlabel('Time [hours]', fontsize=12)
    ax[0].set_ylabel('Stored Electricity\n[GWh]', fontsize=12)
    ax[0].set_title('(a)', fontsize=12)
    ax[0].fill_between(time, 0, original_np, color="black")

    ax[1].set_xlabel('Frequency [hours$^{-1}$]', fontsize=12)
    ax[1].set_ylabel('Magnitude', fontsize=12)
    ax[1].set_title('(b)', fontsize=12)
    ax[1].plot(frequency_np, magnitude_np, color="black")

    ax[2].set_xlabel('Frequency [hours$^{-1}$]', fontsize=12)
    ax[2].set_ylabel('Magnitude', fontsize=12)
    ax[2].set_xlim([-0.005,0.1])
    ax[2].set_title('(c)', fontsize=12)
    ax[2].plot(frequency_np, magnitude_clean_np, color="black")

    offset = 0
    for category in taxonomy:
        if category != "longterm":
            ax[2].plot(frequency_np + offset,frequency_df[category].to_numpy(), color=colours[category])
            offset+=frequency_np[15]

    ax[3].set_xlabel('Time [hours]', fontsize=12)
    ax[3].set_ylabel('Stored Electricity\n[GWh]', fontsize=12)
    ax[3].set_title('(d)', fontsize=12)

    cumsum = np.zeros(len(energy_df["intraday"].to_numpy()))
    for i in range(0,len(taxonomy)):
        cumsum_new = cumsum + energy_df[taxonomy[i]].to_numpy()
        if i == 0:
            ax[3].fill_between(time, 0, cumsum_new, color=colours[taxonomy[i]])
        else:
            ax[3].fill_between(time, cumsum, cumsum_new, color=colours[taxonomy[i]])
        cumsum = cumsum_new

    ax[3].legend(taxonomy, loc='lower center', fontsize=12,frameon=False, bbox_to_anchor=(0.5, -0.9), ncol=5)

    plt.tight_layout()
    plt.show()  
    
def plot2():
    capacity_df = pd.read_csv("./Capacities.csv")

    taxonomy = ["longterm","seasonal","monthly","overnight","intraday"]
    colours = {
        "longterm": "tab:blue",
        "seasonal": "tab:orange",
        "monthly": "tab:green",
        "overnight": "tab:red",
        "intraday": "tab:purple"
    }
    
    demand_np = capacity_df["Demand [MWh/capita/year]"]
    capacity_df = capacity_df.drop(columns=["Demand [MWh/capita/year]"])
    proportion_df = capacity_df.copy()

    for i in range(0,21):
        total_capacity = sum(capacity_df.iloc[i])

        for category in taxonomy:
            proportion_df[category].iloc[i] = 100 * capacity_df[category].iloc[i] / total_capacity
    
    print(total_capacity)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    for category in taxonomy:
        ax[0].plot(demand_np,capacity_df[category].to_numpy(), color=colours[category])
        ax[1].plot(demand_np,proportion_df[category].to_numpy(), color=colours[category])

    ax[0].set_title('(a)', fontsize=11)
    ax[0].set_ylabel('Required Energy\nCapacity [GWh]', fontsize=12)
    ax[0].set_xticks([])

    ax[1].set_title('(b)', fontsize=11)
    ax[1].set_xlabel('Demand [MWh/capita/year]', fontsize=12)
    ax[1].set_ylabel('Proportion of Energy\nCapacity [%]', fontsize=12)

    ax[1].legend(taxonomy, loc='lower center', fontsize=11,frameon=False, bbox_to_anchor=(0.5, -0.7), ncol=5)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    plot1()