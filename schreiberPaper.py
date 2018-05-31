'''
	Replication of the original results from Schreiber et. al (2010)
'''

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from joblib import Parallel, delayed
import multiprocessing
from infoPy import *
import sys

def tentMap(x_in):
	return x_in * 2.0 * (x_in < 0.5) + (2 - 2 * x_in) * (x_in >= 0.5)

def ulamMap(x_in):
	return 2.0 - x_in**2

def pairTE(binMapValues, M, i):
		if i == 0:
			return binTransferEntropy(binMapValues[:,0], binMapValues[:,M-1], 1)
		else:
			return binTransferEntropy(binMapValues[:,i], binMapValues[:,i-1], 1)

def simulateMap(f_map, coupling = 0.05, M = 100, T = 100000):

	values = np.random.rand(M)  # Initialize with random values

	# Run 10^5 steps transient
	for t in range(1, T):
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

	binMapValues = (mapValues >= 0.5).astype(int)
	return binMapValues

exp = sys.argv[-1]

if __name__ == '__main__':

	if exp == 'tent':

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
		for coupling in couplings:
			print 'Coupling = ' + str(coupling)
			binMapValues = simulateMap(tentMap, coupling = coupling, M = M, T = T)

			TEvalues = Parallel(n_jobs=4)( delayed(pairTE)(binMapValues, M, i) for i in range(1, M) )	
			TEmean.append( np.mean(TEvalues) )
			TEstd.append( np.std(TEvalues) / float( len(TEvalues)**0.5 ) )

		plt.figure()
		plt.errorbar(couplings, TEmean, TEstd)
		# Data generated with JIDT toolbox for comparation
		jidt = pd.read_csv('tentMap_jidt.dat', delimiter=',', header=None, names=['c', 'te', 'ste'])
		plt.errorbar(jidt['c'], jidt['te'], jidt['ste'])
		plt.title('Transfer Entropy for the Tent Map')
		plt.ylabel('TE [bits]')
		plt.xlabel(r'$\epsilon$')
		plt.legend(['InfoPy TE', 'JIDT TE'])
		#plt.savefig('TE_TENTMAP.pdf', dpi = 600)

	if exp == 'ulam':
		None