'''
	Python module to compute information theoretical quantities
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial as ss
from math import log,pi,exp

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

def KSGcountings(x, distances):
	'''
	Description: Computes the mean of the digamma functions of the countings as described in Kraskov et. al. (2004), 
	see Eq. 8 in the original paper.
	Inputs:
	X: List with values with one signal
	disatnces: vector of distances (see method KSGestimator_Multivariate)
	Output:
	avg: Mean digamma functions of the countings.
	'''
	from scipy.special import digamma

	N = len(x)
	tree = ss.cKDTree(x)
	avg  = 0
	for i in range(N):
		Nc = len(tree.query_ball_point(x[i], distances[i] - 1e-15, p=float('inf')))
		if Nc > 0:
			avg += digamma(Nc) / float(N)
	return avg

def KSGestimator_Multivariate(X, k = 3, base = 2):
	'''
	Description: Computes mutual information using the KSG estimator (for more information see Kraskov et. al 2004).
	Inputs:
	X: Array with the signals.
	k: Number of nearest neighbors.
	base: Log base (2 for unit bits)
	Output:
	I / log(base): Mutual information (if base=2 in bits, if base=e in nats) 
	'''
	from scipy.special import digamma

	N = len(X[0])
	m = len(X)
	Z = ZIP(X)
	tree = ss.cKDTree(Z)
	# Find nearest neighbors in joint space, p=inf means max-norm
	distances = [tree.query(z, k + 1, p=float('inf'))[0][k] for z in Z]
	I = 0
	for i in range(m):
		I -= KSGcountings(X[i], distances)
	I = I + digamma(k) + (m-1)*digamma(N)
 	return I / log(base)

def ZIP(X):
	'''
	Description: Its the same as the python's zip method, but for lists of arrays.
	Inputs:

	'''
	N = len(X[0])
	C = len(X)
	zipped = []
	for i in range(N):
		aux = []
		for j in range(C):
			aux.append(X[j][i][0])
		zipped.append(aux)
	return zipped