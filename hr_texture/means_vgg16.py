import os
import tensorflow as tf
import numpy as np
import time
import inspect


data = np.load("./vgg19.npy", encoding='latin1').item()
#for k, v in data.items():
    
#    print(k, v)

keys = data.keys()
print(len(keys))

print(keys)
"""
for i in keys:
    value = data[i]
    print(len(value))
    print(":-)")
"""

a = data['conv3_1']
print(data['conv3_1'])
print(type(a))
print(len(a))
print(a[0])
print(np.shape(a[0]))
print(type(a[0]))
print("***************************************")
print(a[1])
print(np.shape(a[1]))

for k in data.keys():
    for i in range(2):
        tmp = np.shape(data[k][i])
        print(tmp)
        l = len(tmp)
        print(k,i)
        if i == 0:
            if l > 2:
                data[k][i] = np.random.rand(tmp[0],tmp[1], tmp[2], tmp[3])

            else:
                data[k][i] = np.random.rand(tmp[0],tmp[1])
        else:
            data[k][i] = np.random.rand(tmp[0])
