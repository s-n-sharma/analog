import translator
import Circuit as circ
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


components = translator.parse_netlist("TestCircuit.txt")
circuit = translator.build_circuit(components)

import matplotlib.pyplot as plt
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

plots_dir = Path(__file__).with_suffix('').parent / 'plots'
plots_dir.mkdir(exist_ok=True)

# ----- magnitude plot -----
plt.figure()
plt.semilogx(f, mag)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Sallen‑Key Low‑Pass Filter – Magnitude Response")
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig(plots_dir / 'magnitude_response.png', dpi=300)

# ----- phase plot -----
plt.figure()
plt.semilogx(f, phase)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (deg)")
plt.title("Sallen‑Key Low‑Pass Filter – Phase Response")
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig(plots_dir / 'phase_response.png', dpi=300)

plt.show()   # keeps interactive behaviour