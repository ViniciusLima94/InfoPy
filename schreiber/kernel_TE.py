import sys
sys.path.append('../infoPy')

import numpy             as np 
import matplotlib.pyplot as plt
import pandas            as pd
from   sklearn.neighbors import KernelDensity
from   kde               import *

def KernelEstimatorTE(x, y, bw = 0.3, norm=True):

	# Normalizing data
	if norm == True:
		x = (x - np.mean(x))/np.std(x)
		y = (y - np.mean(y))/np.std(y)

	# Applying delays
	x_t   = x[1:]
	y_tm1 = y[0:-1]
	x_tm1 = x[0:-1]

	grid = np.vstack([x_t,y_tm1,x_tm1]).T

	pdf_x_t_x_tm1   = box_kernel(np.vstack([x_t, x_tm1]).T, np.vstack([x_t, x_tm1]).T, bw)                   # p(X_t, X_t-1)
	pdf_y_tm1_x_tm1 = box_kernel(np.vstack([y_tm1, x_tm1]).T, np.vstack([y_tm1, x_tm1]).T, bw)               # p(Y_t-1, X_t-1)
	pdf_x_t_y_tm1_x_tm1 = box_kernel(np.vstack([x_t, y_tm1, x_tm1]).T, np.vstack([x_t, y_tm1, x_tm1]).T, bw) # p(X_t, Y_t-1, X_t-1)
	pdf_x_tm1           = box_kernel(x_tm1, x_tm1, bw)                                                       # p(X_t-1)


	H_x_t_x_tm1   = 0
	H_y_tm1_x_tm1 = 0
	H_x_t_y_tm1_x_tm1 = 0
	H_x_tm1           = 0
	for i in range(grid.shape[0]):
		if pdf_x_t_x_tm1[i] > 0:
			H_x_t_x_tm1   -= np.log2(pdf_x_t_x_tm1[i])

		if pdf_y_tm1_x_tm1[i] > 0:
			H_y_tm1_x_tm1 -= np.log2(pdf_y_tm1_x_tm1[i])

		if pdf_x_t_y_tm1_x_tm1[i] > 0:
			H_x_t_y_tm1_x_tm1 -= np.log2(pdf_x_t_y_tm1_x_tm1[i]) 

		if pdf_x_tm1[i] > 0:
			H_x_tm1           -= np.log2(pdf_x_tm1[i])

	TE = ( H_x_t_x_tm1 + H_y_tm1_x_tm1 - H_x_t_y_tm1_x_tm1 - H_x_tm1 ) / grid.shape[0] 
	return TE

# Read in the data
rawData = pd.read_csv('data.txt', header = None, delimiter = ',')
x = rawData[0].values # Extracts what Matlab does with 2350:3550 argument there.
# Chest vol is second column
y = rawData[1].values

TE12 = KernelEstimatorTE(y, x, bw = 0.032, norm=True)
TE21 = KernelEstimatorTE(x, y, bw = 0.032, norm=True)