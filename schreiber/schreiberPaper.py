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

exp = sys.argv[-1]  # Which result to run

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
			binMapValues = simulateMap(tentMap, coupling = coupling, M = M, Transient = T, T = T)

			TEvalues = Parallel(n_jobs=40)( delayed(pairTE)(binMapValues, M, i) for i in range(1, M) )	
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
		def ulamMI(coupling):
			print 'Coupling = ' + str(coupling)
                        x1, x2 = simulateMap(ulamMap, coupling = coupling, M = 100, Transient = 100000, T = 10000)
                        idx = np.arange(0, len(x1)+100, 100)
                        MI12 = KernelEstimatorMI(x1, x2, bw = 0.3, norm=False, delay = 1)
                        MI21 = KernelEstimatorMI(x2, x1, bw = 0.3, norm=False, delay = 1)
                        TE12 = KernelEstimatorTE(x1, x2, bw = 0.3, norm=False)
                        TE21 = KernelEstimatorTE(x2, x1, bw = 0.3, norm=False)
			return np.array([MI12, MI21, TE12, TE21])

		M = 100
		Transient = 100000
		T = 10000
		couplings = np.arange(0, 1.02, 0.02)
		MI = np.squeeze( Parallel(n_jobs=40)( delayed(ulamMI)(coupling) for coupling in couplings ) )
		plt.plot(couplings, MI[:, 0], '--')
		plt.plot(couplings, MI[:, 1], '--')
		plt.plot(couplings, MI[:, 2])
		plt.plot(couplings, MI[:, 3])
		plt.xlim([0, 1])
		plt.ylim([0, 2.5])
		plt.show()

	if exp == 'ulam-data':

		def ulamData(count):
			print 'Coupling = ' + str(count)
			data = np.loadtxt('ulam-data/ulam_'+str(count)+'.dat', delimiter=',')
			MI12 =  0#KSGestimatorMI(data[:, 0], data[:, 1], k = 3, norm = False, noiseLevel = 1e-8)#KernelEstimatorMI(data[:, 0], data[:, 1], kernel = 'gaussian', bw = 0.3, norm=False, delay = 1)
			MI21 =  0#KSGestimatorMI(data[:, 1], data[:, 0], k = 3, norm = False, noiseLevel = 1e-8)#KernelEstimatorMI(data[:, 1], data[:, 0], kernel = 'gaussian', bw = 0.3, norm=False, delay = 1)
			TE12 =  KSGestimatorTE(data[:, 0], data[:, 1], k = 3, norm = False, noiseLevel = 1e-8)#KernelEstimatorTE(data[:, 0], data[:, 1], kernel = 'gaussian', bw = 0.3, norm=False) 
			TE21 =  KSGestimatorTE(data[:, 1], data[:, 0], k = 3, norm = False, noiseLevel = 1e-8)#KernelEstimatorTE(data[:, 1], data[:, 0], kernel = 'gaussian', bw = 0.3, norm=False) 
			return np.array([MI12, MI21, TE12, TE21])

		couplings = np.arange(0, 1.02, 0.02)

		data = np.squeeze( Parallel(n_jobs=40)( delayed(ulamData)(count) for count in range(0, 51) ) )

		#plt.plot(couplings, data[:, 0], '--')
		#plt.plot(couplings, data[:, 1], '--')
		plt.plot(couplings, data[:, 2])
		plt.plot(couplings, data[:, 3])

	if exp == 'hb':
		# Read in the data
		rawData = pd.read_csv('data.txt', header = None, delimiter = ',')
		x = rawData[0].values # Extracts what Matlab does with 2350:3550 argument there.
		# Chest vol is second column
		y = rawData[1].values
		rs = np.array([0.01, 0.016, 0.023, 0.032, 0.047, 0.064, 0.09, 0.12, 0.18, 0.26, 0.36, 0.50, 0.65, 1.0])
		MI12 = np.zeros(len(rs))
		MI21 = np.zeros(len(rs))
		TE12 = np.zeros(len(rs))
		TE21 = np.zeros(len(rs))
		for i in range( len(rs) ):
			print 'r = ' + str(rs[i])
			MI12[i] = KernelEstimatorMI(x, y, bw = rs[i], kernel = 'gaussian', delay = 1, norm=True) 
			MI21[i] = KernelEstimatorMI(y, x, bw = rs[i], kernel = 'gaussian', delay = 1, norm=True) 
			TE12[i] = KernelEstimatorTE(x, y, bw = rs[i], kernel = 'gaussian', norm=True)
			TE21[i] = KernelEstimatorTE(y, x, bw = rs[i], kernel = 'gaussian', norm=True)
		plt.semilogx(rs, MI12, '--')
		plt.semilogx(rs, MI21, '--')
		plt.semilogx(rs, TE12)
		plt.semilogx(rs, TE21, '-.')
		plt.legend(['TE(heart->breath)', 'TE(breath->heart)'])
		plt.xlim([0.01, 1])
		plt.ylim([0, 5])
		plt.ylabel('TE')




