import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image


# Mac users may need device = 'mps' (untested)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


D = torch.load('C.pt')
print(D.shape)



def score_forward(D):
    # D = (timesteps, num_samples, num_channels, width, height) 
    T = D.shape[0]
    num_samples = D.shape[1]

    c = torch.zeros(T)
    d = torch.zeros(T)
    energy = 0

    for j in range(num_samples):
        for i in range(T):
            c[i] = torch.norm(D[i,j,:,:,:])
            
        energy  += torch.norm(D[0,j,:,:,:])
        d += c

    d = d / (energy)
  
    return d

def score_backward(D):
    # D = (timesteps, num_samples, num_channels, width, height) 
    T = D.shape[0]
    num_samples = D.shape[1]

    c = torch.zeros(T)
    d = torch.zeros(T)
    energy = 0

    for j in range(num_samples):
        for i in range(T):
            c[i] = torch.norm(D[i,j,:,:,:])
            
        energy  += torch.norm(D[-1,j,:,:,:])
        d += c

    d = d / (energy)
  
    return d

d = score_forward(D)
e = score_backward(D)


plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
plt.figure(figsize=(8,6), dpi=50)
plt.plot(d, lw=4)
plt.xlabel(' Time Steps', fontsize=20)
plt.ylabel('Score Function', fontsize=20)
#plt.xlim([0,T+1])
#plt.ylim([0,1])
plt.grid()
plt.savefig('score3.pdf')
#torch.save(d, 'score_cosine.pt')






plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
plt.figure(figsize=(8,6), dpi=50)
plt.plot(e, lw=4)
plt.xlabel(' Time Steps', fontsize=20)
plt.ylabel('Score Function', fontsize=20)
#plt.xlim([0,T+1])
#plt.ylim([0,1])
plt.grid()
plt.savefig('score_2.pdf')
#torch.save(d, 'score_cosine.pt')
