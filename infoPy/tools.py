'''
	Replication of the original results from Schreiber et. al (2010)
'''

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from joblib import Parallel, delayed
import multiprocessing
from discrete import *
import sys

def tentMap(x_in):
	r'''
	Description: Tent Map as described in Schreiber (2000)
	Inputs:
	x_in: Input or previous value of the map
	Outputs:
	Next value of the map.
	'''
	return x_in * 2.0 * (x_in < 0.5) + (2 - 2 * x_in) * (x_in >= 0.5)

def ulamMap(x_in):
	r'''
	Description: Ulam Map as described in Schreiber (2000)
	Inputs:
	x_in: Input or previous value of the map
	Outputs:
	Next value of the map.
	'''
	return 2.0 - x_in**2

def pairTE(binMapValues, M, i):
		r'''
		Description: Quantifie TE for each pair of elements of the tent map.
		Inputs:
		binMapValues: Binary map output.
		M: Number of elements in the map
		i: Element index.
		Outputs:
		Transfer enetropy of thepair i;i-1.
		'''
		if i == 0:
			return binTransferEntropy(binMapValues[:,0], binMapValues[:,M-1], 1)
		else:
			return binTransferEntropy(binMapValues[:,i], binMapValues[:,i-1], 1)

def simulateMap(f_map, coupling = 0.05, M = 100, Transient = 100000, T = 100000):
	r'''
	Description: Simulate maps.
	Inputs:
	f_map: Map function
	coupling: Counpling between the map elements
	M: Number of elements in the map
	Transiente: Transient time
	T: Simulation time
	Output
	if f_map = tentMap; The binary output of each map element
	if f_map = ulamMap; The output of the first two elements of the map
	'''
	'''
		Initializing map values.
		For the tent map it's initialized with random uniform values.
		For the ulam map with random uniform values un the range [-2, 2]
	'''
	if f_map == tentMap:
		values = np.random.rand(M)  
	elif f_map == ulamMap:
		values = np.random.uniform(-2, 2, size = M)

	# Run 10^5 steps transient
	for t in range(1, Transient):
		for i in range(len(values)):
			if i == 0:
				values[i] = f_map( coupling * values[-1] + (1 - coupling) * values[i])
			else:
				values[i] = f_map( coupling * values[i-1] + (1 - coupling) * values[i])


	mapValues = np.zeros([T, M])
	mapValues[0, :] = values

	# Run 10^5 steps simulation
	for t in range(1, T):
		for m in range(0, M):
			if m == 0:
				mapValues[t, m] = f_map( coupling * mapValues[t-1, -1] + (1 - coupling) * mapValues[t-1, m] )
			else:
				mapValues[t, m] = f_map( coupling * mapValues[t-1, m-1] + (1 - coupling) * mapValues[t-1, m] )

	if f_map == tentMap:
		binMapValues = (mapValues >= 0.5).astype(int)
		return binMapValues
	elif f_map == ulamMap:
		x1 = mapValues[:, 0]
		x2 = mapValues[:, 1]
		return x1, x2