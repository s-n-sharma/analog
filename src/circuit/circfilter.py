import numpy as np
import random
import os
import matplotlib.pyplot as plt

import math

#size = 5#define each block to be of like 5 data points

def compute_randomness(arr, size):
    ret = []
    for i in range(int(len(arr)/size)):
        avg = sum(arr[5*i:5*(i+1)])/size
        ret.extend([arr[5*i] - avg, arr[5*i + 1] - avg, arr[5*i + 2] - avg, arr[5*i + 3] - avg, arr[5*i + 4] - avg])
    return np.dot(ret, ret)/np.dot(arr, arr)

DIR = 'ex_circ_mag'

count = 0

for filename in os.listdir(DIR):
    if not os.path.isfile(os.path.join(DIR, filename)):
        continue
    file = os.path.join(DIR, filename)
    data = []
    with open (file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(float(line.split()[-1]))
    data = np.array(data)
    if compute_randomness(data, 5) > 0.03:
        count += 1
        f = np.logspace(1, 6, 400)
        plt.figure()
        plt.semilogx(f, data)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Sallen‑Key Low‑Pass Filter – Magnitude Response")
        plt.grid(True, which='both')
        plt.show()

print(count)







"""

rand_arr = np.random.rand(400)
lin = np.array([math.sin(50*x) for x in np.linspace(-1, 1, 400)])
ra = rand_arr
every_5 = []
lin5 = []

for i in range(int(len(rand_arr)/size)):
    avg = sum(rand_arr[5*i:5*(i+1)])/size
    avg2 = sum(lin[5*i:5*(i+1)])/size
    every_5.extend([ra[5*i] - avg, ra[5*i + 1] - avg, ra[5*i + 2] - avg, ra[5*i + 3] - avg, ra[5*i + 4] - avg])
    lin5.extend([lin[5*i] - avg2, lin[5*i + 1] - avg2, lin[5*i + 2] - avg2, lin[5*i + 3] - avg2, lin[5*i + 4] - avg2])

every_5 = np.array(every_5)
lin5 = np.array(lin5)
print(np.dot(every_5, every_5)/np.dot(rand_arr, rand_arr))
print(np.dot(lin5, lin5)/np.dot(lin, lin))

"""


