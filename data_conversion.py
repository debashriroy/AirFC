import numpy as np
from scipy.io import loadmat,savemat
# import os
import ipdb; ipdb.set_trace()

model_inputs = loadmat('model_inputs_16.mat')
out = model_inputs['out_x'][0:100]  # original data out
air_out = np.zeros((100,64,10), dtype=np.complex64) # OTA out

for i in range(0,100):
    ex = loadmat('data\\example'+str(i)+'.mat')
    air_out[i] = ex['out_symbols']

print(air_out.shape)
print(out.shape)
# print(air_out)



