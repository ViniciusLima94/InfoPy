import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from infoPy import *
from tools import *
import sys

idx = int(sys.argv[-1])

def pairTE(binMapValues, M, i):
	if i == 0:
		return binTransferEntropy(binMapValues[:,0], binMapValues[:,M-1], 1)
	else:
		return binTransferEntropy(binMapValues[:,i], binMapValues[:,i-1], 1)

M = 100
T = 100000

couplings = np.arange(0, 0.052, 0.002)

TEmean = []
TEstd  = []

binMapValues = simulateMap(tentMap, coupling = couplings[idx], M = M, Transient = T, T = T)

TE = []
for i in range(1, M):
	TE.append( pairTE(binMapValues, M, i) )

TEmean = np.mean(TE) 
TEstd  = np.std(TE) / np.sqrt(M) 

np.save('data_maps/tent/tent_'+str(idx)+'.npy', {'TEmean': TEmean, 'TEstd': TEstd})

