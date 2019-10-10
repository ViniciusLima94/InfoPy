'''
	Python module to compute information theoretical quantities
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial as ss
from math import log,pi,exp
from sklearn.neighbors import NearestNeighbors

def KSGestimatorMI(x, y, k = 3, norm = True, noiseLevel = 1e-8):
	'''
	Description: Computes mutual information using the KSG estimator (for more information see Kraskov et. al 2004).
	Inputs:
	x, y: Array with the signals.
	k: Number of nearest neighbors.
	base: Log base (2 for unit bits)
	norm: Whether to normalize or not the data
	noiseLevel: Level of noise added to the data to break degeneracy
	Output:
	I: Mutual information.
	'''
	from scipy.special import digamma
	from sklearn.neighbors import NearestNeighbors

	N = len(x)

	# Add noise to the data to break degeneracy
	x = x + 1e-8*np.random.rand(N)
	y = y + 1e-8*np.random.rand(N)

	# Normalizing data
	if norm == True:
		x = (x - np.mean(x))/np.std(x)
		y = (y - np.mean(y))/np.std(y)

	Z = np.squeeze(np.array([x[:, None], y[:, None]]).T)
	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', metric='chebyshev').fit(Z)
	distances, _ = nbrs.kneighbors(Z)
	distances = distances[:, k]

	nx = np.zeros(N)
	ny = np.zeros(N)

	for i in range(N):
		nx[i] = np.sum( np.abs(x[i]-x) < distances[i] )
		ny[i] = np.sum( np.abs(y[i]-y) < distances[i] )
	I = digamma(k) - np.mean( digamma(nx) + digamma(ny) ) + digamma(N)
	return I

def KSGestimatorMImultivariate(X, k = 3, norm = True, noiseLevel = 1e-8):
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
	from sklearn.neighbors import NearestNeighbors

	N = len(X[0])
	m = len(X)

	for i in range(m):
		X[i] = X[i] + 1e-8*np.random.rand(N, 1)

	if norm == True:
		for i in range(m):
			X[i] = (X[i] - np.mean(X[i]))/np.std(X[i])

	Z = np.squeeze( ZIP(X) )
	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', metric='chebyshev').fit(Z)
	distances, _ = nbrs.kneighbors(Z)
	distances = distances[:, k]

	n = np.zeros([N, m])

	for j in range(m):
		for i in range(N):
			n[i][j] = np.sum( np.abs(X[j][i]-X[j]) < distances[i] )

	I = digamma(k) + (m-1) * digamma(N)

	for i in range(m):
		I -= np.mean( digamma(n[:,i]) )

	return I

def delayedKSGMI(x, y, k = 3, norm = True, noiseLevel = 1e-8, delay = 0):
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
	return KSGestimator_Multivariate(x, y, k = k, norm = norm, noiseLevel = noiseLevel)

def KSGestimatorTE(x, y, k = 3, norm = True, noiseLevel = 1e-8):
	'''
	Description: 
	Inputs:
	x, y: Array with the signals.
	k: Number of nearest neighbors.
	base: Log base (2 for unit bits)
	norm: Whether to normalize or not the data
	noiseLevel: Level of noise added to the data to break degeneracy
	Output:
	I: Mutual information.
	'''
	from scipy.special import digamma
	from sklearn.neighbors import NearestNeighbors

	Ni = len(x)

	# Normalizing data
	if norm == True:
		x = (x - np.mean(x))/np.std(x)
		y = (y - np.mean(y))/np.std(y)

	# Add noise to the data to break degeneracy
	x = x + 1e-8*np.random.rand(Ni)
	y = y + 1e-8*np.random.rand(Ni)

	# Applying shifts
	ym = y[1:]
	x  = x[:-1]
	y  = y[:-1]		

	N = len(x)

	
	Z = np.squeeze( ZIP([x[:, None], y[:, None], ym[:, None]]) )
	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', metric='chebyshev').fit(Z)
	distances, _ = nbrs.kneighbors(Z)
	distances = distances[:, k]

	nx = np.zeros(N)
	ny = np.zeros(N)
	nym = np.zeros(N)

	for i in range(N):
		nx[i] = np.sum( np.sqrt( (y[i]-y)**2 + (x[i]-x)**2 ) < distances[i] )
		ny[i] = np.sum( np.sqrt( (y[i]-y)**2 + (ym[i]-ym)**2 ) < distances[i] )
		nym[i] = np.sum( np.abs(ym[i]-ym) < distances[i] )
		
	I = digamma(k) + np.mean( digamma(nym) - digamma(ny) - digamma(nx) )
	return I
	

##################################################################################################
# AUXILIARY FUNCTIONS                                                                            #
##################################################################################################

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth,metric='chebyshev',algorithm='ball_tree', **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def kde_estimator(x, y, x_grid, y_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    #data       = np.concatenate( (x[:, None], y[:, None]) , axis = 1)
    #data_grid  = np.concatenate( (x_grid[:, None], y_grid[:, None]) , axis = 1)

    data = np.vstack([x,y]).T
    data_grid = np.vstack([x_grid, y_grid]).T

    kde_skl = KernelDensity(bandwidth=bandwidth,metric='chebyshev',algorithm='ball_tree', **kwargs)
    kde_skl.fit(data)
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(data_grid)
    return np.exp(log_pdf)

def kde_estimator2(x, y, z, x_grid, y_grid, z_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    #data       = np.concatenate( (x[:, None], y[:, None], z[:, None]) , axis = 1)
    #data_grid  = np.concatenate( (x_grid[:, None], y_grid[:, None], z_grid[:, None]) , axis = 1)

    data = np.vstack([x,y,z]).T
    data_grid = np.vstack([x_grid, y_grid, z_grid]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, metric='chebyshev',algorithm='ball_tree', **kwargs)
    kde_skl.fit(data)
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(data_grid)
    return np.exp(log_pdf)

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
