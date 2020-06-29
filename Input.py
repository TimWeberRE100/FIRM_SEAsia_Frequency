# Modelling input and assumptions
# Copyright (c) 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from Optimisation import percapita, node

Nodel = np.array(['AW', 'AN', 'BN', 'KH', 'CN', 'IN', 'IJ', 'IK', 'IM', 'IP', 'IC', 'IS', 'IT', 'LA', 'MY', 'MM', 'PL', 'PM', 'PV', 'SG', 'TH', 'VH', 'VS'])
PVl =   np.array(['BN']*1 + ['KH']*1 + ['IJ']*1 + ['IK']*1 + ['IM']*1 + ['IP']*1 + ['IC']*1 + ['IS']*1 + ['IT']*1 + ['LA']*1 + ['MY']*1 + ['MM']*1 + ['PL']*1 + ['PM']*1 + ['PV']*1 + ['SG']*1 + ['TH']*1 + ['VH']*1 + ['VS']*1)
Windl = np.array(['KH']*1 + ['LA']*1 + ['MM']*1 + ['PL']*1 + ['PM']*1 + ['PV']*1 + ['TH']*1 + ['VH']*1 + ['VS']*1)
Interl = np.array(['AW']*1 + ['AN']*1 + ['CN']*1 + ['IN']*1)
resolution = 1

MLoad = np.genfromtxt('Data/electricity{}.csv'.format(percapita), delimiter=',', skip_header=1) # EOLoad(t, j), MW
baseload = np.genfromtxt('Data/baseload.csv', delimiter=',', skip_header=1) # EBLoad(0, j), MW
TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1) # TSPV(t, i), MW
TSWind = np.genfromtxt('Data/wind.csv', delimiter=',', skip_header=1) # TSWind(t, i), MW
CHydro = np.genfromtxt('Data/hydro.csv', delimiter=',', skip_header=1) * pow(10, -3) # CHydro(j), MW to GW

inter = 0.05 if node=='Super2' else 0
CDC0max, CDC1max, CDC7max, CDC8max = 4 * [inter * MLoad.sum() / MLoad.shape[0] / 1000] # 5%: AWIJ + ANIT, CHVH, INMM, MW to GW
DCloss = np.array([2100, 1000, 900, 1300, 1300, 500, 200, 600, 1000, 900, 1400, 2100, 900, 600, 1000, 1000, 500, 500, 300, 1300, 700, 600, 400]) * 0.03 * pow(10, -3)

if node in ['BN', 'SG']:
    efficiency = 0.9
    factor = np.genfromtxt('Data/factor1.csv', delimiter=',', usecols=1)
else:
    efficiency = 0.8
    factor = np.genfromtxt('Data/factor.csv', delimiter=',', usecols=1)

firstyear, finalyear, timestep = (2010, 2019, 1)

if 'Super' not in node:
    MLoad = MLoad[:, np.where(Nodel==node)[0]]
    baseload = baseload[np.where(Nodel==node)[0]]
    TSPV = TSPV[:, np.where(PVl==node)[0]]
    TSWind = TSWind[:, np.where(Windl==node)[0]]
    CHydro = CHydro[np.where(Nodel==node)[0]]

if node!='Super2':
    Interl = np.array([])

intervals, nodes = MLoad.shape
years = int(resolution * intervals / 8760)
pzones, wzones = (TSPV.shape[1], TSWind.shape[1])
pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + nodes)
inters = len(Interl)
iidx = sidx + 1 + inters

energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW
CHydro1 = CHydro - baseload * pow(10, -3) # MW to GW
Hydromax = baseload.sum() * 8760 * 2 # Max contribution from hydro and other renewables, TWh to MWh p.a.

manage = 4 # weeks
allowance = MLoad.sum(axis=1).max() * 0.05 * manage * 168 * efficiency # MWh

class Solution:
    """A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)"""

    def __init__(self, x):
        self.x = x
        self.MLoad = MLoad
        self.intervals, self.nodes = (intervals, nodes)
        self.resolution = resolution
        self.baseload = baseload

        self.CPV = list(x[: pidx]) # CPV(i), GW
        self.CWind = list(x[pidx: widx]) # CWind(i), GW
        self.GPV = TSPV * np.tile(self.CPV, (intervals, 1)) * pow(10, 3) # GPV(i, t), GW to MW
        self.GWind = TSWind * np.tile(self.CWind, (intervals, 1)) * pow(10, 3) # GWind(i, t), GW to MW

        self.CPHP = list(x[widx: sidx]) # CPHP(j), GW
        self.CPHS = x[sidx] # S-CPHS(j), GWh
        self.efficiency = efficiency

        self.CInter = x[sidx+1: iidx] if node=='Super2' else [0] # CInter(j), GW
        self.GInter = np.tile(self.CInter, (intervals, 1)) * pow(10, 3) # GInter(j, t), GW to MW

        self.Nodel, self.PVl, self.Windl, self.Interl = (Nodel, PVl, Windl, Interl)
        self.node = node

        self.CHydro, self.CHydro1 = (CHydro, CHydro1)

        self.allowance = allowance

    def __repr__(self):
        """S = Solution(list(np.ones(64))) >> print(S)"""
        return 'Solution({})'.format(self.x)