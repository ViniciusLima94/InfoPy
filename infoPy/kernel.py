'''
	Python module to compute information theoretical quantities
'''

import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd
import scipy.spatial     as ss
import os
from   math import log,pi,exp
from   sklearn.neighbors import NearestNeighbors
from   sklearn.neighbors import KernelDensity
from   jpype             import *

def KernelEstimatorMI(x, y, bw = 0.3, kernel = 'tophat', delay = 0, norm = True):
	r'''
	Description: Computes the mutual information between two signals x and y.
	Inputs:
	x: Signal x.
	y: Signal y.
	bw: bandwidth of the kernel estimator.
	kernel: Kernel used in the KDE estimator ('gaussian', 'tophat', 'cosine'; see http://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html)
	delay: Delay applied between x and y, for the delayed mutual information
	norm: Sets whether the data will be normalized or not.
	Outputs:
	MI: Returns the mutual information between x and y.
	'''

	# Normalizing data
	if norm == True:
		x = (x - np.mean(x))/np.std(x)
		y = (y - np.mean(y))/np.std(y)

	# Applying delays
	if delay == 0:
		x = x
		y = y
	elif delay > 0:
		x = x[:-delay]
		y = y[delay:]

	N = len(x)

	grid  = np.vstack([x, y])

	pdf_x = kde_sklearn(x, grid[0], kernel = kernel, bandwidth=bw)
	pdf_y = kde_sklearn(y, grid[1], kernel = kernel, bandwidth=bw)
	pdf_xy = kde_estimator(y, x, grid[1], grid[0], kernel = kernel, bandwidth=bw)

	MI = 0
	count = 0
	for i in range(len(pdf_x)):
		if pdf_x[i] > 0.00001 and pdf_y[i] > 0.00001 and pdf_xy[i] > 0.00001:
			MI += np.log2( pdf_xy[i] / ( pdf_x[i]*pdf_y[i] ) ) 
			count += 1.0
	return MI / count

def KernelEstimatorMImulti(x, y, z, bw = 0.3, kernel = 'tophat', norm = True):
	r'''
	Description: Computes the mutual information between two signals x and y.
	Inputs:
	x: Signal x.
	y: Signal y.
	bw: bandwidth of the kernel estimator.
	kernel: Kernel used in the KDE estimator ('gaussian', 'tophat', 'cosine'; see http://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html)
	delay: Delay applied between x and y, for the delayed mutual information
	norm: Sets whether the data will be normalized or not.
	Outputs:
	MI: Returns the mutual information between x and y.
	'''

	# Normalizing data
	if norm == True:
		x = (x - np.mean(x))/np.std(x)
		y = (y - np.mean(y))/np.std(y)
		z = (z - np.mean(z))/np.std(z)

	N = len(x)

	grid  = np.vstack([x, y, z])

	pdf_x = kde_sklearn(x, grid[0], kernel = kernel, bandwidth=bw)
	pdf_y = kde_sklearn(y, grid[1], kernel = kernel, bandwidth=bw)
	pdf_z = kde_sklearn(z, grid[2], kernel = kernel, bandwidth=bw)
	pdf_xy = kde_estimator(x, y, grid[0], grid[1], kernel = kernel, bandwidth=bw)
	pdf_xz = kde_estimator(x, z, grid[0], grid[2], kernel = kernel, bandwidth=bw)
	pdf_yz = kde_estimator(y, z, grid[1], grid[2], kernel = kernel, bandwidth=bw)
	pdf_xyz = kde_estimator2(x, y, z, grid[0], grid[1], grid[2], kernel = kernel, bandwidth=bw)

	MI = 0
	count = 0
	for i in range(len(pdf_x)):
		if pdf_x[i] > 0.00001 and pdf_y[i] > 0.00001 and pdf_z[i] > 0.00001 and pdf_xy[i] > 0.00001 and pdf_xz[i] > 0.00001 and pdf_yz[i] > 0.00001 and pdf_xyz[i] > 0.00001:
			MI += np.log2( pdf_xy[i]*pdf_xz[i]*pdf_yz[i] / (pdf_xyz[i]*pdf_x[i]*pdf_y[i]*pdf_z[i]) ) 
			count += 1.0
	return MI / count


def KernelEstimatorTE(x, y, bw = 0.3, norm=True):
	r'''
	Description: Computes the transfer entropy between two signals X and Y, defined by
	TE(X->Y) = I(X_t:Y_t-1|X_t-1) = H(X_t,X_t-1)+H(Y_t-1,X_t-1)-H(X_t,Y_t-1,X_t-1)-H(X_t-1) 
	(equation 4.4 and 4.31 in An introduction to transfer entropy: Information flow in complex systems)
	Inputs:
	x: Signal X.
	y: Signal Y.
	bw: bandwidth of the kernel estimator.
	kernel: Kernel used in the KDE estimator ('gaussian', 'tophat', 'cosine'; see http://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html)
	delay: Delay applied between x and y, for the delayed transfer entropy
	norm: Sets whether the data will be normalized or not.
	Outputs:
	TE: Returns the transfer entropy from x to y.
	'''

	# Normalizing data
	if norm == True:
		x = (x - np.mean(x))/np.std(x)
		y = (y - np.mean(y))/np.std(y)

	# Applying delays
	x_t   = y[1:]
	y_tm1 = x[0:-1]
	x_tm1 = y[0:-1]

	N = len(x_t)

	grid = np.vstack([x_t,y_tm1,x_tm1]).T

	#N = 50
	#xmin, xmax = x.min(), x.max()
	#ymin, ymax = y.min(), y.max()

	#p1 = np.linspace(xmin, xmax, N) 
	#p2 = np.linspace(ymin, ymax, N)  
	#p3 = np.linspace(xmin, xmax, N)  

	#grid=np.meshgrid(p1, p2, p3)         
	#grid=np.reshape(grid, (3,N**3)).T

	#kde = KDEmultivariate(x_t, bw=bandwidth * np.ones_like(x), vartype='c', kernel='uni')
	#pdf_x_t_x_tm1 = kde.evaluate(x_grid)

	'''
	pdf_x_t_x_tm1   = kde_estimator(x_t, x_tm1, grid[:,0], grid[:,2], kernel = kernel, bandwidth=bw)                        # p(X_t, X_t-1)
	pdf_y_tm1_x_tm1 = kde_estimator(y_tm1, x_tm1, grid[:,1], grid[:,2], kernel = kernel, bandwidth=bw)                      # p(Y_t-1, X_t-1)
	pdf_x_t_y_tm1_x_tm1 = kde_estimator2(x_t, y_tm1, x_tm1, grid[:,0], grid[:,1], grid[:,2], kernel = kernel, bandwidth=bw) # p(X_t, Y_t-1, X_t-1)
	pdf_x_tm1           = kde_sklearn(x_tm1, grid[:,2], kernel = kernel, bandwidth=bw)                                      # p(X_t-1)

	
	box_kernel(np.vstack([x_t, x_tm1]).T, np.vstack([x_t, x_tm1]).T, bw)#
	box_kernel(np.vstack([y_tm1, x_tm1]).T, np.vstack([y_tm1, x_tm1]).T, bw)#
	box_kernel(np.vstack([x_t, y_tm1, x_tm1]).T, np.vstack([x_t, y_tm1, x_tm1]).T, bw)#
	box_kernel(x_tm1, x_tm1, bw)#
	'''

	pdf_x_t_x_tm1   = KernelDensityEstimator(np.vstack([x_t, x_tm1]).T, 2, bw)
	pdf_y_tm1_x_tm1 = KernelDensityEstimator(np.vstack([y_tm1, x_tm1]).T, 2, bw) 
	pdf_x_t_y_tm1_x_tm1 = KernelDensityEstimator(np.vstack([x_t, y_tm1, x_tm1]).T, 3, bw)
	pdf_x_tm1           = KernelDensityEstimator(x_tm1, 1, bw)

	H_x_t_x_tm1   = np.zeros(N)
	H_y_tm1_x_tm1 =  np.zeros(N)
	H_x_t_y_tm1_x_tm1 =  np.zeros(N)
	H_x_tm1           =  np.zeros(N)
	for i in range(N):
		if pdf_x_t_x_tm1[i] > 0:
			H_x_t_x_tm1[i]   = -np.log2(pdf_x_t_x_tm1[i])

		if pdf_y_tm1_x_tm1[i] > 0:
			H_y_tm1_x_tm1[i] = -np.log2(pdf_y_tm1_x_tm1[i])

		if pdf_x_t_y_tm1_x_tm1[i] > 0:
			H_x_t_y_tm1_x_tm1[i] = -np.log2(pdf_x_t_y_tm1_x_tm1[i]) 

		if pdf_x_tm1[i] > 0:
			H_x_tm1[i]           = -np.log2(pdf_x_tm1[i])

	TE = ( H_x_t_x_tm1.mean() + H_y_tm1_x_tm1.mean() - H_x_t_y_tm1_x_tm1.mean() - H_x_tm1.mean() ) 
	return TE

##################################################################################################
# AUXILIARY FUNCTIONS                                                                            #
##################################################################################################
def KernelDensityEstimator(X, d, bandwidth):
	r'''
		Computes the KDE for uni and multivariable, i'm using the algorithm by Lizier which uses
		a java backend
		Inputs:
		X : Data matrix, must be Nobservations x Ndimensions
	'''

	jarLocation = os.path.join('infodynamics.jar')
	if isJVMStarted() == False:
		startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

	if d == 1:
		kernel = JPackage('infodynamics.measures.continuous.kernel').KernelEstimatorUniVariate
		kernel = kernel()
		kernel.setNormalise(False)
		kernel.initialise(bandwidth)
		kernel.setObservations(X)
		p = np.array( [kernel.getProbability(obs) for obs in X] )

	else:
		kernel = JPackage('infodynamics.measures.continuous.kernel').KernelEstimatorMultiVariate
		kernel = kernel()
		kernel.setNormalise(False)
		kernel.initialise(d, bandwidth)
		kernel.setObservations(X)
		p = np.array( [kernel.getProbability(obs) for obs in X] )

	return p

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth,algorithm='kd_tree', **kwargs)
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

    kde_skl = KernelDensity(bandwidth=bandwidth,algorithm='kd_tree', **kwargs)
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

    kde_skl = KernelDensity(bandwidth=bandwidth,algorithm='kd_tree', **kwargs)
    kde_skl.fit(data)
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(data_grid)
    return np.exp(log_pdf)
