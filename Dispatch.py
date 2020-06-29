# Step-by-step analysis to decide the dispatch of hydropower and other renewables
# Copyright (c) 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from Input import *
from Simulation import Reliability

import datetime as dt
from multiprocessing import Pool, cpu_count

def Flexible(instance):
    """Energy source of high flexibility"""

    year, x = instance
    print('Dispatch works on', year)

    S = Solution(x)

    # startidx = int((24 / resolution) * (dt.datetime(year, 1, 1) - dt.datetime(firstyear, 1, 1)).days)
    # endidx = int((24 / resolution) * (dt.datetime(year+1, 1, 1) - dt.datetime(firstyear, 1, 1)).days)

    startidx = int((24 / resolution) * (year - firstyear) * 365)
    endidx = int((24 / resolution) * (year+1 - firstyear) * 365)

    HBcapacity = CHydro.sum() * pow(10, 3) # GW to MW
    hydro = HBcapacity * np.ones(endidx - startidx)

    for i in range(0, endidx - startidx, timestep):
        hydro[i: i+timestep] = baseload.sum()
        Deficit = Reliability(S, hydro=hydro, start=startidx, end=endidx) # Sj-EDE(t, j), MW
        if Deficit.sum() * resolution - S.allowance > 0.1:
            hydro[i: i+timestep] = HBcapacity

    hydro = np.clip(hydro - S.Spillage, baseload.sum(), None)
    contribution = hydro.sum() * resolution * pow(10, -6) # MWh to TWh
    print('Hydro & other renewables contribution in %s (TWh):' % year, contribution)

    return hydro

def Analysis(x):
    """Dispatch.Analysis(result.x)"""

    starttime = dt.datetime.now()
    print('Dispatch starts at', starttime)

    # Multiprocessing
    pool = Pool(processes=min(cpu_count(), finalyear - firstyear + 1))
    instances = map(lambda y: [y] + [x], range(firstyear, finalyear + 1))
    Dispresult = pool.map(Flexible, instances)
    pool.terminate()

    Hydro = np.concatenate(Dispresult)
    np.savetxt('Results/Dispatch_Hydro{}{}.csv'.format(node, percapita), Hydro, fmt='%f', delimiter=',', newline='\n', header='Hydro & other renewables')

    endtime = dt.datetime.now()
    print('Dispatch took', endtime - starttime)

    from Statistics import Information
    Information(x, Hydro)

    return True

if __name__ == '__main__':
    capacities = np.genfromtxt('Data/Optimisation_resultx.csv', delimiter=',', skip_header=1)
    Analysis(capacities)