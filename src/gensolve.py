from scipy.cluster.hierarchy import DisjointSet as djs
import translator 
import Circuit as circ
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os


DIR = "example_circuits"

for filename in os.listdir(DIR):
    try:
        if not os.path.isfile(os.path.join(DIR, filename)):
            continue
        file = os.path.join(DIR, filename)
        components = translator.parse_netlist(file)
        circuit = translator.build_circuit(components)

        f = np.logspace(1, 10, 400)
        mag = np.zeros_like(f, dtype=float)
        phase = np.zeros_like(f, dtype=float)

        for i, freq in enumerate(f):
            omega = 2 * np.pi * freq
            circuit.setFrequency(omega)
            V = circuit.solveSystem()
            if V is None:
                continue
            Vout = V[circuit.VOUT]
            mag[i] = 20 * np.log10(np.abs(Vout))
            phase[i] = np.angle(Vout, deg=True)
        
        for i in range(len(mag)):
            with open("ex_circ_mag/"+filename, 'a') as out:
                out.write(f'{f[i]} {mag[i]}\n')
            with open("ex_circ_phase/"+filename, 'a') as out:
                out.write(f'{f[i]} {mag[i]}\n')
    except Exception as e:
        print(str(e))



