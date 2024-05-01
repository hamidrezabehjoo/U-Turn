import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image


# Mac users may need device = 'mps' (untested)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


D = torch.load('A.pt')
print(D.shape)

def forward_corr(D):
    # D is the input tensor (timesteps, samples, channels, height, width)
    num_samples = D.shape[1]
    T = D.shape[0]

    c = torch.zeros(T)
    d = torch.zeros(T)
    energy = 0

    for j in range(num_samples):
        for i in range(T):
            c[i] = torch.sum( D[0,j,:,:,:] * D[i,j,:,:,:])
        energy  += torch.sum( D[0,j,:,:,:] * D[0,j,:,:,:])
        d += c

    d = d / (energy)
    
    return d


def reverse_corr(D):
    # D is the input tensor (timesteps, samples, channels, height, width)
    num_samples = D.shape[1]
    T = D.shape[0]

    c = torch.zeros(T)
    d = torch.zeros(T)
    energy = 0

    for j in range(num_samples):
        for i in range(T):
            c[i] = torch.sum( D[-1,j,:,:,:] * D[i,j,:,:,:])
        energy  += torch.sum( D[-1,j,:,:,:] * D[-1,j,:,:,:])
        d += c

    d = d / (energy)

    return d


d = forward_corr(D)
T = D.shape[0] 

plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
plt.figure(figsize=(8,6))
plt.plot(d, lw=4)
plt.xlabel(' Time Steps', fontsize=20)
plt.ylabel('Forward Correlation', fontsize=20)
plt.xlim([0,T])
#plt.ylim([0,1])
plt.grid()
#plt.xscale('log')
plt.savefig('a.pdf')
#torch.save(d, 'c1L.pt')



e = reverse_corr(D) 
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
plt.figure(figsize=(8,6))
plt.plot(e, lw=4)
plt.xlabel(' Time Steps', fontsize=20)
plt.ylabel('Reverse Correlation', fontsize=20)
plt.xlim([0,T])
#plt.ylim([0,1])
plt.grid()

plt.savefig('b.pdf')
#torch.save(d, 'c2L.pt')




