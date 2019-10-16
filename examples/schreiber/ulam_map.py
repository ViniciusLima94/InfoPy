import sys
sys.path.append('../../infoPy')

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from   kernel import KernelEstimatorDelayedMI, KernelEstimatorTE
from   tools import *

def ulamData_kernel(count):
	print('Coupling = ' + str(count))
	data = np.loadtxt('data_maps/ulam/ulam_'+str(count)+'.dat', delimiter=',')
	MI12 =  0#KernelEstimatorDelayedMI(data[:, 0], data[:, 1], bw = 0.3, delay = 1, norm=False)
	MI21 =  0#KernelEstimatorDelayedMI(data[:, 1], data[:, 0], bw = 0.3, delay = 1, norm=False)
	TE12 =  KernelEstimatorTE(data[:, 0], data[:, 1], bw = 0.3, delay = 1, norm=False) 
	TE21 =  KernelEstimatorTE(data[:, 1], data[:, 0], bw = 0.3, delay = 1, norm=False) 
	return np.array([MI12, MI21, TE12, TE21])

for i in range(1, 52):
	data_kernel = ulamData_kernel(i)
	np.save('data_maps/ulam/ulam_kernel_'+str(i)+'.npy', {'MI12':data_kernel[0], 'MI21':data_kernel[1], 'TE12':data_kernel[2], 'TE21':data_kernel[3]})


# Plotting data
couplings = np.arange(0, 1.02, 0.02)

MI12 = np.zeros(couplings.shape[0])
MI21 = np.zeros(couplings.shape[0])
TE12 = np.zeros(couplings.shape[0])
TE21 = np.zeros(couplings.shape[0])

for i in range(1,couplings.shape[0]+1):
	data = np.load('data_maps/ulam/ulam_kernel_'+str(i)+'.npy').item()
	MI12[i-1] = data['MI12']
	MI21[i-1] = data['MI21']
	TE12[i-1] = data['TE12']
	TE21[i-1] = data['TE21']

plt.plot(couplings, MI12, '--')
plt.plot(couplings, MI21, '--')
plt.plot(couplings, TE12)
plt.plot(couplings, TE21)
plt.xlabel('Coupling')
plt.ylabel('TE [bits]')
plt.title('ULAM map TE')
plt.legend(['MI(1->2)', 'MI(2->1)', 'TE(1->2)', 'TE(2->1)'])
plt.savefig('figures/ulam_te_kernel.png', dpi=600)
plt.close()