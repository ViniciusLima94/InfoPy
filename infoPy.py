'''
	Python module to compute information theoretical quantities
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial as ss
from math import log,pi,exp
from sklearn.neighbors import NearestNeighbors

def BinNeuronEntropy(SpikeTrain):
	r'''
	Description: Computes the entropy of a binary neuron (with a response consisting of 0s and 1s).
	Inputs:
	SpikeTrain: Binary spike train of a neuron (must be composed of 0s and 1s)
	Outputs:
	Returns the entropy of the binary spike train
	'''
	T = len(SpikeTrain) # Length of the spike train
	P_firing = np.sum(SpikeTrain) / float(T) # Probability of firing (probability of the spike trai being 1)
	P_notfiring = 1.0 - P_firing # Pobability of silence (probability of the spike tain being 0)
	# Computing and returning the entropy of the binary spike train
	return -P_firing*np.log2(P_firing) - P_notfiring*np.log2(P_notfiring)


def EntropyFromProbabilities(Prob):
	r'''
	Description: Computes the entropy of given probability distribution.
	Inputs:
	Prob: Probability distribution of a random variable
	Outputs:
	H: Entropy of the probabilitie distribution.
	'''
	H = 0
	s = Prob.shape
	for p in Prob:
		if p > 0.00001:
			H -= p*np.log2(p)
	return H

def binMutualInformation(sX, sY, tau):
	r'''
	Description: Computes the delayed mutual information bewtween two binary spike trains.
	Inputs:
	sX: Binary spike train of neuron X
	sY: Binary spike train of neuron Y
	tau: Delay applied in the spike train of neuron Y
	Outputs:
	MI: Returns the mutual information MI(sX, sY)
	'''
	PX = np.zeros([2])
	PY = np.zeros([2])
	PXY = np.zeros([2,2])
	for t in range( np.maximum(0, 0-tau), np.minimum(len(sX)-tau, len(sX)) ):
		PX[1] += sX[t]
		PY[1] += sY[t+tau]
		if (sX[t]==0) and (sY[t+tau]==0):
			continue
		else:
			PXY[sX[t], sY[t+tau]] += 1

	# Estimating [0, 0] pairs for the PXY matrix
	N = len(sX)   # Number of bins in the spike train
	Np = N - tau  # Number of pairs
	PXY[0,0] = Np - np.sum(PXY)
	PX[0]    = Np - PX[1]
	PY[0]    = Np - PY[1]
	# Normalizing probabilities
	PX = PX / np.sum(PX)
	PY = PY / np.sum(PY)
	PXY = PXY / np.sum(PXY)
	HX = EntropyFromProbabilities(PX)
	HY = EntropyFromProbabilities(PY)
	HXY = EntropyFromProbabilities(np.reshape(PXY, (4)))
	MI  = HX + HY - HXY
	return MI

def binTransferEntropy(x, y, delay):
	r'''
	Description: Computes the delayed transfer entropy, from y to x, bewtween two binary spike trains.
	Inputs:
	x: Binary spike train of neuron X
	y: Binary spike train of neuron Y
	delay: Delay applied in the spike train of neuron Y
	Outputs:
	TE: Returns the Transfer Entropy TEy->x
	'''
	T = len(x)

	if delay == 1:
		delay = 0
	elif delay > 1:
		delay -= 1

	px = np.array( [T-np.sum(x), np.sum(x)] ) / float(T) # p(x)
	py = np.array( [T-np.sum(y), np.sum(y)] ) / float(T) # p(y)
	pxy = np.zeros([2, 2])                               # p(x, y)
	pxy1 = np.zeros([2, 2])                              # p(x1, y)
	pxyz = np.zeros([2, 2, 2])                           # p(x1, x, y)

	for i in range(0, T-delay):
		if (x[i+delay] == 0 and y[i] == 0):
			continue
		else:
			pxy[ x[i+delay], y[i] ] += 1.0

	pxy[0,0] = (T - delay - np.sum(pxy))
	pxy = pxy / float(T-delay)

	for i in range(0, T-1):
		if (x[i+1] == 0 and x[i] == 0):
			continue
		else:
			pxy1[ x[i+1], x[i] ] += 1.0

	for i in range(0, T-1-delay):
		if (x[i+1+delay] == 0 and x[i+delay] == 0 and y[i] == 0):
			continue
		else:
			pxyz[ x[i+1+delay], x[i+delay], y[i] ] += 1.0

	pxy1[0,0] = (T - 1 - np.sum(pxy1))
	pxyz[0,0,0] = (T - 1 - delay - np.sum(pxyz))
	# Normalizing probabilities
	pxy1 = pxy1 / float(T-1)
	pxyz = pxyz / float(T-1-delay)

	TE = 0
	for xn in [0, 1]:
		for yn in [0, 1]:
			for xn1 in [0, 1]:
				if pxy1[xn1, xn] > 0.00001 and pxy[xn, yn] > 0.00001 and pxyz[xn1, xn, yn] > 0.00001 and px[xn] > 0.00001: 
					TE += pxyz[xn1, xn, yn] * np.log2( ( pxyz[xn1, xn, yn] * px[xn] ) / ( pxy1[xn1, xn] * pxy[xn, yn] ) )
				else:
					continue
	return TE


def KSGestimator_Multivariate(x, y, k = 3, norm = True, noiseLevel = 1e-8):
	'''
	Description: Computes mutual information using the KSG estimator (for more information see Kraskov et. al 2004).
	Inputs:
	X: Array with the signals.
	k: Number of nearest neighbors.
	base: Log base (2 for unit bits)
	norm: Whether to normalize or not the data
	noiseLevel: Level of noise added to the data to break degeneracy
	Output:
	I: Mutual information.
	'''
	from scipy.special import digamma

	N = len(x)

	# Add noise to the data to break degeneracy
	x = x + 1e-8*np.random.randn(N)
	y = y + 1e-8*np.random.randn(N)

	# Normalizing data
	if norm == True:
		x = (x - np.mean(x))/np.std(x)
		y = (y - np.mean(y))/np.std(y)

	distances = np.zeros(N)
	for i in range(0, N):
		d = []
		for j in range(0, N):
			dX, dY = np.abs(x[i]-x[j]), np.abs(y[i]-y[j])
			d.append( max( dX, dY ) )
		distances[i] = np.sort(d)[k]

 	nx = np.zeros(N)
 	ny = np.zeros(N)

 	for i in range(N):
 		nx[i] = np.sum( np.abs(x[i]-x) < distances[i] )
 		ny[i] = np.sum( np.abs(y[i]-y) < distances[i] )
 	I = digamma(k) - np.mean( digamma(nx) + digamma(ny)  ) + digamma(N)
 	return I

def delayedKSGMI(x, y, k = 3, delay = 0):
	'''
	Description: Computes the delayed mutual information using the KSG estimator (see method KSGestimator_Multivariate).
	Inputs:
	X: Array with the signals.
	k: Number of nearest neighbors.
	base: Log base (2 for unit bits)
	delay: Delay applied
	Output:
	I / log(base): Mutual information (if base=2 in bits, if base=e in nats) 
	'''
	if delay == 0:
		x = x
		y = y
	elif delay > 0:
		x = x[:-delay]
		y = y[delay:]
	return KSGestimator_Multivariate(x, y, k = k)
