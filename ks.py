import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import scipy.stats as stats




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


D = torch.load('A.pt')
print(D.shape)


T, N, w, h, c = D.shape 
A_vec = torch.reshape(D, (T, N, w * h * c)) # vectorize the tensor
print(A_vec.shape)

dim = w * h * c 

b = torch.zeros(T)
k = 0

for j in range(0,T):
    sample = A_vec[j, :, :]
    a = torch.zeros(dim)

    for i in range(dim):
        x = sample[:, i]
        #ks, p = stats.ks_1samp(x, stats.norm.cdf)
        ks, p = stats.ks_1samp(x, stats.norm.cdf, args=(x.mean(), x.std()))
        if ks > 0.06:
            #print(ks, p)
            a[i] = 1
    
    b[k] = torch.count_nonzero(a)
    #print( torch.count_nonzero(a) )
    
    if j%10 == 0:
        print(k, ks, b[k]/dim)
    k += 1
    

plt.plot(b/dim)
plt.savefig('ks.pdf')

    
    
