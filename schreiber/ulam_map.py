import sys
sys.path.append('../infoPy')

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from kernel import *
from tools import *

#idx = int(sys.argv[-1])
'''
def ulamData_ksg(count):
	print('Coupling = ' + str(count))
	data = np.loadtxt('data_maps/ulam/ulam_'+str(count)+'.dat', delimiter=',')
	MI12 =  KSGestimatorMI(data[:, 0], data[:, 1], k = 1, norm = False, noiseLevel = 1e-8)
	MI21 =  KSGestimatorMI(data[:, 1], data[:, 0], k = 1, norm = False, noiseLevel = 1e-8)
	TE12 =  KSGestimatorTE(data[:, 0], data[:, 1], k = 1, norm = False, noiseLevel = 1e-8) 
	TE21 =  KSGestimatorTE(data[:, 1], data[:, 0], k = 1, norm = False, noiseLevel = 1e-8) 
	return np.array([MI12, MI21, TE12, TE21])
'''
def ulamData_kernel(count):
	print('Coupling = ' + str(count))
	data = np.loadtxt('data_maps/ulam/ulam_'+str(count)+'.dat', delimiter=',')
	#MI12 =  KernelEstimatorMI(data[:, 0], data[:, 1], bw = 0.3, norm=False, delay = 1)
	#MI21 =  KernelEstimatorMI(data[:, 1], data[:, 0], bw = 0.3, norm=False, delay = 1)
	TE12 =  KernelEstimatorTE(data[:, 0], data[:, 1], bw = 0.3, norm=False) 
	TE21 =  KernelEstimatorTE(data[:, 1], data[:, 0], bw = 0.3, norm=False) 
	return np.array([TE12, TE21])

for i in range(1, 52):
	data_kernel = ulamData_kernel(i)
	#data_ksg    = ulamData_ksg(idx)

	np.save('data_maps/ulam/ulam_kernel_'+str(i)+'.npy', {'TE12':data_kernel[0], 'TE21':data_kernel[1]})
	#np.save('data_maps/ulam/ulam_ksg_'+str(idx)+'.npy', {'M12':data_ksg[0], 'M21':data_ksg[1], 'TE12':data_ksg[2], 'TE21':data_ksg[3]})
