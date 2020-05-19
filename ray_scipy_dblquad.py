import numpy as np
import scipy.integrate as scint
import time
import ray

ray.init(memory=102005473280,object_store_memory=21474836480)
start = time.time()

@ray.remote
def f(th, ph):
	return np.sin(th)/(4*np.pi)
nth=91
nphi=181

theta=np.linspace(0,np.pi,nth)
phi=np.linspace(0,2*np.pi,nphi)
Area=[]
conv_fac=180.0/np.pi
for iphi in range(nphi-1):
	for ith in range(nth-1):
		print('-----------------------------------------------')
		print('Phi = ' + str(conv_fac*phi[iphi]))
		print('Theta = ' + str(conv_fac*theta[ith]))
		f=lambda theta, phi : np.sin(theta)/(4*np.pi)
		[A,err]=scint.dblquad(f,phi[iphi], phi[iphi+1],lambda phi: theta[ith], lambda phi: theta[ith+1],epsabs=1e-6)
		print(A)
		Area.append(A)
print('-----------------------------------------------')
end=time.time()
#Area_full=(Area)
#print(Area)
print('Area = ' + str(sum(Area)))
total_time=end-start

print('Total time taken = ' + str(total_time) +' seconds')
