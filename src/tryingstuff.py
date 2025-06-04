
import numpy as np
import matplotlib.pyplot as plt
import Circuit as circuit
import math
import random
circ = circuit.Circuit()

Vin = circuit.VoltageSource(1.0)  
op  = circuit.IdealOpAmp()
R1  = circuit.Resistor(5e3)
R2  = circuit.Resistor(7e1)

C1  = circuit.Capacitor(1e-9)
C2  = circuit.Capacitor(1e-9)
for c in (Vin, op, R1, R2, C1, C2):
    circ.addComponent(c)


circ.connectComponents(Vin, 'p', R1, 'n')
circ.connectComponents(Vin, 'n', None, None)    
circ.connectComponents(R1, 'p', R2, 'n')         
circ.connectComponents(R2, 'p', C2, 'p')        
circ.connectComponents(C2, 'n', None, None)      
circ.connectComponents(R2, 'n', op, 'V+')     
circ.connectComponents(R2, 'n', C1, 'n')       
circ.connectComponents(C2, 'p', op, 'Vout')      
circ.connectComponents(op, 'V-', op, 'Vout') 



f = np.logspace(1, 5, 400)           
mag = np.zeros_like(f, dtype=float)
phase = np.zeros_like(f, dtype=float)

for i, freq in enumerate(f):
    omega = 2 * np.pi * freq
    circ.setFrequency(omega)
    V = circ.solveSystem()
    Vout = V[op.Vout]
    mag[i] = 20 * np.log10(np.abs(Vout))
    phase[i] = np.angle(Vout, deg=True)


plt.figure()
plt.semilogx(f, mag)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Sallen‑Key Low‑Pass Filter – Magnitude Response")
plt.grid(True, which='both')


plt.figure()
plt.semilogx(f, phase)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (deg)")
plt.title("Sallen‑Key Low‑Pass Filter – Phase Response")
plt.grid(True, which='both')
plt.show()


################################################################################################################

def compute_bode(circ, f, mag, phase):
    for i, freq in enumerate(f):
        omega = 2 * np.pi * freq
        circ.setFrequency(omega)
        V = circ.solveSystem()
        Vout = V[op.Vout]
        mag[i] = 20 * np.log10(np.abs(Vout))
        phase[i] = np.angle(Vout, deg=True)
    return mag, phase

def compute_loss(a, b, c):
    mag = a[0]
    phase = a[1]
    return np.dot(mag-b, mag-b)+np.dot(phase -c, phase - c)

Vin = circuit.VoltageSource(1.0)  
op  = circuit.IdealOpAmp()
R1  = circuit.Resistor(10e3)
R2  = circuit.Resistor(10e3)

C1  = circuit.Capacitor(10e-9)
C2  = circuit.Capacitor(10e-9)


circ = circuit.Circuit()

for c in (Vin, op, R1, R2, C1, C2):
    circ.addComponent(c)

circ.connectComponents(Vin, 'p', R1, 'n')
circ.connectComponents(Vin, 'n', None, None)    
circ.connectComponents(R1, 'p', R2, 'n')         
circ.connectComponents(R2, 'p', C2, 'p')        
circ.connectComponents(C2, 'n', None, None)      
circ.connectComponents(R2, 'n', op, 'V+')     
circ.connectComponents(R2, 'n', C1, 'n')       
circ.connectComponents(C2, 'p', op, 'Vout')      
circ.connectComponents(op, 'V-', op, 'Vout') 

#x' = x - eta * dL/dx

#dL/dx = [L(x + epsilon) - L(x)]/epsilon
#epsilon = nu * x

mag_p = np.zeros_like(f, dtype=float)
phase_p = np.zeros_like(f, dtype=float)
nu = 0.00001
eta = 0.00001

k = 0

# plt.figure()
# plt.semilogx(f, mag)
# plt.semilogx(f, compute_mag(circ, f, mag_p))
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude (dB)")
# plt.title("Sallen‑Key Low‑Pass Filter – Magnitude Response")
# plt.grid(True, which='both')

# plt.show()

print(f"Initial parameters: {compute_loss(compute_bode(circ, f, mag_p, phase_p), mag, phase)}")

max_change_factor = 0.1 


while (k < 100):
    changes = []
    original_L = compute_loss(compute_bode(circ, f, mag_p, phase_p), mag, phase)
    for comp in circ.components:
        if isinstance(comp, circuit.Resistor) or isinstance(comp, circuit.Capacitor):
            original_val = comp.value
            epsilon = max(nu * original_val, 0.000000000001)
            comp.value += epsilon
            perturbed_L =  compute_loss(compute_bode(circ, f, mag_p, phase_p), mag, phase)
            dLdX = (perturbed_L - original_L)/epsilon
            change = -1 * eta * dLdX
            comp.value = original_val
            if abs(change) > abs(max_change_factor * comp.value):
                change = math.copysign(max_change_factor * comp.value, change)
            changes.append(change)
        else:
            changes.append('N/A')
    
    for i in range(len(circ.components)):
        if not changes[i] == 'N/A':
            circ.components[i].value = max(1e-10, circ.components[i].value + changes[i])
         
    if (k % 50 == 0):
        print(f"Training Round {k}: {compute_loss(compute_bode(circ, f, mag_p, phase_p), mag, phase)}")
    k += 1

mag_p, phase_p = compute_bode(circ, f, mag_p, phase_p)
    

plt.figure()
plt.semilogx(f, mag)
plt.semilogx(f, mag_p)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Sallen‑Key Low‑Pass Filter – Magnitude Response")
plt.grid(True, which='both')

plt.figure()
plt.semilogx(f, phase)
plt.semilogx(f, phase_p)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (deg)")
plt.title("Sallen‑Key Low‑Pass Filter – Phase Response")
plt.grid(True, which='both')

plt.show()

for comp in circ.components:
    if isinstance(comp, circuit.Resistor) or isinstance(comp, circuit.Capacitor):
        print(comp.value)
            



#phase_p = np.zeros_like(f, dtype=float)




############################################