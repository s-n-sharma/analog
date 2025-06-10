from scipy.cluster.hierarchy import DisjointSet as djs
import translator
import Circuit as circ
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="raise")
import os


DIR = "example_circuits"
f = np.logspace(1, 6, 400)


for filename in os.listdir(DIR):
    if not os.path.isfile(os.path.join(DIR, filename)):
        continue
    file = os.path.join(DIR, filename)
    trans = translator.Translator(file)
    circuit = trans.circuit

    mag = np.zeros_like(f, dtype=float)
    phase = np.zeros_like(f, dtype=float)

    failed_flag = False

    try:

        for i, freq in enumerate(f):
            omega = 2 * np.pi * freq
            circuit.setFrequency(omega)
            V = circuit.solveSystem()
            if V is None or V == "failed":
                failed_flag = True
                break
            Vout = V[circuit.VOUT]
            mag[i] = 20 * np.log10(np.abs(Vout))
            phase[i] = np.angle(Vout, deg=True)
   
    except Exception as e:
        #print(str(e))
        continue

    if failed_flag:
        continue
    
    for i in range(len(mag)):
        with open("ex_circ_mag/"+filename, 'a') as out:
            out.write(f'{f[i]} {mag[i]}\n')
        with open("ex_circ_phase/"+filename, 'a') as out:
            out.write(f'{f[i]} {phase[i]}\n')




