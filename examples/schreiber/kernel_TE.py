import sys
sys.path.append('../infoPy')

import numpy             as np 
import matplotlib.pyplot as plt
import pandas            as pd
from   sklearn.neighbors import KernelDensity
import os
from   jpype             import *

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


jarLocation = os.path.join('infodynamics.jar')
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

kernel_1d = JPackage('infodynamics.measures.continuous.kernel').KernelEstimatorUniVariate

data = np.loadtxt('data_maps/ulam/ulam_1.dat', delimiter=',')
x    = data[:,0]
y    = data[:,1]
# Applying delays
x_t   = y[1:]
y_tm1 = x[0:-1]
x_tm1 = y[0:-1]

bw = 0.3
kernel = 'tophat'

X = np.vstack([x_t,y_tm1,x_tm1]).T

k=10
j=10
p1 = np.linspace(-2,2, 14) # Array size k : J
p2 = np.linspace(-2,2, 14) # Array size k : J
p3 = np.linspace(-2,2, 14) # Array size k : J

grid=np.meshgrid(p1, p2,p3) # Meshigrid
pars=np.reshape(grid, (3,14**3)).T # Every pair possible formed by p1 and p2

'''
pdf_x_t_x_tm1   = kde_estimator(x_t, x_tm1, x_t, x_tm1, kernel = kernel, bandwidth=bw, metric='euclidean')                                # p(X_t, X_t-1)
pdf_y_tm1_x_tm1 = kde_estimator(y_tm1, x_tm1, y_tm1, x_tm1, kernel = kernel, bandwidth=bw, metric='euclidean')                          # p(Y_t-1, X_t-1)
pdf_x_t_y_tm1_x_tm1 = kde_estimator2(x_t, y_tm1, x_t, x_t, y_tm1, x_t, kernel = kernel, bandwidth=bw, metric='euclidean')   # p(X_t, Y_t-1, X_t-1)
pdf_x_tm1           = kde_sklearn(x_tm1, x_tm1, kernel = kernel, bandwidth=bw, metric='euclidean')                                        # p(X_t-1)
'''

kernel = JPackage('infodynamics.measures.continuous.kernel').KernelEstimatorUniVariate
kernel = kernel()
kernel.setNormalise(False)
kernel.initialise(0.3)
kernel.setObservations(JArray(JDouble, 1)(x_tm1))
pdf_x_tm1 = np.array( [kernel.getProbability(obs) for obs in x_tm1] )

H_x_t_x_tm1   = np.zeros_like(x_t)
H_y_tm1_x_tm1 = np.zeros_like(x_t)
H_x_t_y_tm1_x_tm1 = np.zeros_like(x_t)
H_x_tm1           = np.zeros_like(x_t)

for i in range(x_t.shape[0]):
	if pdf_x_t_x_tm1[i] > 0:
		H_x_t_x_tm1[i] = -np.log2(pdf_x_t_x_tm1[i])

	if pdf_y_tm1_x_tm1[i] > 0:
		H_y_tm1_x_tm1[i] = -np.log2(pdf_y_tm1_x_tm1[i])  

	if pdf_x_t_y_tm1_x_tm1[i] > 0:
		H_x_t_y_tm1_x_tm1[i] = -np.log2(pdf_x_t_y_tm1_x_tm1[i]) 

	if pdf_x_tm1[i] > 0:
		H_x_tm1[i] = -np.log2(pdf_x_tm1[i])

TE =  H_x_t_x_tm1.mean() + H_y_tm1_x_tm1.mean() - H_x_t_y_tm1_x_tm1.mean() - H_x_tm1.mean()

print('H(X_t-1) = ' +str(np.round(H_x_tm1.mean(),4)) + ' bits')
print('H(Y_t-1, X_t-1) = ' + str(np.round(H_y_tm1_x_tm1.mean(),4)) + ' bits')
print('H(X_t, X_t-1) = ' + str(np.round(H_x_t_x_tm1.mean(),4)) + ' bits')
print('H(X_t, Y_t-1, X_t-1) = ' +str(np.round(H_x_t_y_tm1_x_tm1.mean(),4)) + ' bits')
print('TE = ' + str(np.round(TE,4)))
