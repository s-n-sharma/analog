"""
Draw the transfer function on graph, press s and then escape (just ignore the save thing that comes up) and then close the window, and 
close the other windows that come up. It tries to find the circuit by using 
some example circuits (RC high pass and low pass, sallen-key band pass, band-reject),
 finding which one to use, and then optimizing parameters to minimize loss. terminates when loss starts to increase
"""
import numpy as np
import Circuit as circuit
import math
import random
import sys
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from scipy.interpolate import interp1d

class InteractiveLogDraw:
    def __init__(self, x_sample_points):
        self.fig, self.ax = plt.subplots()
        self.x_sample_points = x_sample_points
        self.drawn_points = []
        self.is_drawing = False
        self.sampled_data = [] # Initialize sampled_data

        # Setup the plot
        self.ax.set_xscale('log')
        self.ax.set_xlim(10, 1e6)
        self.ax.set_ylim(-100, 30)
        self.ax.set_xlabel("X (log scale)")
        self.ax.set_ylabel("Y (linear scale)")
        self.ax.set_title("Draw on the semilogx graph. Press 's' to sample.")
        self.line, = self.ax.plot([], [], 'r-', label="Drawn Line")
        self.sampled_line, = self.ax.plot([], [], 'bo', markersize=3, label="Sampled Points (400)")
        self.ax.legend(loc="upper left")

        # Connect events
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        print("Click and drag to draw on the plot.")
        print("Press 's' key to sample the drawn graph at the 400 specified points.")
        print("Press 'c' key to clear the drawing and sampled points.")
        plt.show()

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        
        # --- Critical check for valid coordinates ---
        if event.xdata is None or event.ydata is None or \
           np.isnan(event.xdata) or np.isnan(event.ydata):
            print("Warning: Invalid coordinates on press (None or NaN). Ignoring press.")
            return

        self.is_drawing = True
        # Clip xdata to axis limits; also ensures x_coord is not NaN if event.xdata was finite
        x_coord = np.clip(event.xdata, self.ax.get_xlim()[0], self.ax.get_xlim()[1])
        
        # If x_coord became NaN after clipping (e.g. if axis limits were NaN, highly unlikely)
        if np.isnan(x_coord):
            print("Warning: Clipped x_coord is NaN on press. Ignoring press.")
            return

        # Store the raw ydata, actual y-clipping for interpolation happens later if needed
        self.drawn_points = [(x_coord, event.ydata)]
        
        # For visual feedback, plot with y clipped to view
        y_preview = np.clip(event.ydata, self.ax.get_ylim()[0], self.ax.get_ylim()[1])
        self.line.set_data([x_coord], [y_preview])
        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        if not self.is_drawing or event.inaxes != self.ax:
            return
        
        # --- Critical check for valid coordinates ---
        if event.xdata is None or event.ydata is None or \
           np.isnan(event.xdata) or np.isnan(event.ydata):
            # print("Warning: Invalid coordinates on motion (None or NaN). Ignoring point.") # Can be spammy
            return
        
        x_coord = np.clip(event.xdata, self.ax.get_xlim()[0], self.ax.get_xlim()[1])

        if np.isnan(x_coord): # Should not happen if event.xdata was not NaN
            # print("Warning: Clipped x_coord is NaN on motion. Ignoring point.")
            return
            
        # Ensure x_coord is positive for log scale (clip should handle xlim[0] > 0)
        if self.ax.get_xscale() == 'log' and x_coord <= 0:
            # print(f"Warning: x_coord {x_coord} invalid for log scale on motion. Ignoring point.")
            return

        self.drawn_points.append((x_coord, event.ydata)) # Store raw ydata
        
        # For visual feedback, plot with y clipped to view
        if self.drawn_points:
            preview_x, preview_y_original = zip(*self.drawn_points)
            preview_y_clipped = np.clip(preview_y_original, self.ax.get_ylim()[0], self.ax.get_ylim()[1])
            self.line.set_data(preview_x, preview_y_clipped)
            self.fig.canvas.draw_idle()

    def on_release(self, event):
        if event.button == 1 and self.is_drawing:
            self.is_drawing = False
            
            if len(self.drawn_points) > 1:
                # Sort points by x-value first
                self.drawn_points.sort(key=lambda p: p[0])
                
                # Filter for unique x-values AND ensure no NaNs are in the points for interpolation
                unique_finite_points = []
                if self.drawn_points:
                    last_x = -np.inf # Initialize last_x to ensure the first valid point is included
                    for p_x, p_y in self.drawn_points:
                        # --- Ensure coordinates are finite for interpolation ---
                        if np.isnan(p_x) or np.isnan(p_y) or np.isinf(p_x) or np.isinf(p_y):
                            print(f"Warning: Non-finite coordinate ({p_x:.2e}, {p_y:.2f}) found during processing. Skipping.")
                            continue
                        
                        if p_x > last_x: # Ensures strictly increasing x for interp1d
                            unique_finite_points.append((p_x, p_y))
                            last_x = p_x
                        elif p_x == last_x:
                            # If x is the same, update y with the latest point (optional, or take average, etc.)
                            # For simplicity here, we keep the first y for a given x after sorting.
                            # To use the last y: check if unique_finite_points is not empty and update last element.
                            # unique_finite_points[-1] = (p_x, p_y) # this would update to the last y for same x
                            pass # Current logic keeps the first y encountered for a given x.

                self.drawn_points = unique_finite_points
                
                # Update the visual line with the processed (sorted, unique, finite) points
                if self.drawn_points:
                    final_x, final_y_original = zip(*self.drawn_points)
                    final_y_clipped = np.clip(final_y_original, self.ax.get_ylim()[0], self.ax.get_ylim()[1])
                    self.line.set_data(final_x, final_y_clipped)
                else:
                    self.line.set_data([], []) # Clear line if no valid points remain
                self.fig.canvas.draw_idle()
                
                print(f"Drawing segment finished. {len(self.drawn_points)} valid, unique x-points recorded.")
            elif self.drawn_points: # Single valid point drawn
                print("Drawing finished (single valid point). Not enough for interpolation.")
            else: # No valid points recorded or all filtered out
                print("No valid points recorded for drawing.")


    def on_key_press(self, event):
        if event.key == 's':
            self.sample_drawing()
        elif event.key == 'c':
            self.clear_drawing()

    def clear_drawing(self):
        self.drawn_points = []
        self.line.set_data([], [])
        self.sampled_line.set_data([], [])
        self.sampled_data = [] # Clear sampled data
        self.fig.canvas.draw_idle()
        print("Drawing and sampled points cleared.")

    def sample_drawing(self):
        if len(self.drawn_points) < 2:
            print("Not enough valid points drawn (need at least 2 with unique finite x-coords) to sample.")
            # Fill sampled_data with NaNs if an empty sample is expected
            self.sampled_data = list(zip(self.x_sample_points, [np.nan] * len(self.x_sample_points)))
            self.sampled_line.set_data([], []) # Clear previous samples if any
            self.fig.canvas.draw_idle()
            return

        drawn_x, drawn_y = zip(*self.drawn_points) # These should now be clean and finite

        # drawn_y[0] and drawn_y[-1] should be finite here.
        fill_value_for_extrapolation = (drawn_y[0], drawn_y[-1])

        interp_func = interp1d(drawn_x, drawn_y, kind='linear',
                               bounds_error=False, fill_value=fill_value_for_extrapolation)

        sampled_y_values_raw = interp_func(self.x_sample_points)

        # Clip y-values to the plot's y-limits.
        # NaNs should not be produced by interp_func if inputs are clean and fill_values are finite.
        min_y_lim, max_y_lim = self.ax.get_ylim()
        sampled_y_values_clipped = np.clip(sampled_y_values_raw, min_y_lim, max_y_lim)
        
        # Final check for NaNs that might have slipped through or were intended (e.g., if x_sample_points had NaNs)
        # However, with finite fill values, interp1d with linear kind on finite data should not produce NaNs.
        nan_indices = np.isnan(sampled_y_values_clipped)
        if np.any(nan_indices):
            print(f"Warning: {np.sum(nan_indices)} NaN values still found in final sampled y-values. This is unexpected.")
            # For debugging, you could print where they occur:
            # for i, x_val in enumerate(self.x_sample_points):
            #     if nan_indices[i]:
            #         print(f"  NaN at x_sample={x_val:.2e}")


        self.sampled_line.set_data(self.x_sample_points, sampled_y_values_clipped)
        self.fig.canvas.draw_idle()

        # Recalculate counts based on potentially cleaner data
        num_interpolated = 0
        num_extrapolated_start = 0
        num_extrapolated_end = 0
        # Ensure drawn_x is not empty for boundary checks
        min_drawn_x, max_drawn_x = drawn_x[0], drawn_x[-1]

        for i, x_s in enumerate(self.x_sample_points):
            y_s = sampled_y_values_clipped[i]
            if np.isnan(y_s): # Skip if it's NaN for counting purposes
                continue

            if x_s < min_drawn_x:
                if np.isclose(y_s, drawn_y[0]):
                    num_extrapolated_start += 1
            elif x_s > max_drawn_x:
                if np.isclose(y_s, drawn_y[-1]):
                    num_extrapolated_end += 1
            else: # Within or at the bounds of drawn_x
                 num_interpolated +=1

        print(f"Sampled at {len(self.x_sample_points)} x-points.")
        print(f"  Interpolated (or at exact drawn points): {num_interpolated} points.")
        if num_extrapolated_start > 0:
            print(f"  Extrapolated using first y-value (left of drawing): {num_extrapolated_start} points.")
        if num_extrapolated_end > 0:
            print(f"  Extrapolated using last y-value (right of drawing): {num_extrapolated_end} points.")
        
        total_accounted_for = num_interpolated + num_extrapolated_start + num_extrapolated_end
        num_still_nan = np.sum(np.isnan(sampled_y_values_clipped))
        if num_still_nan > 0:
             print(f"  Points resulting in NaN in final output: {num_still_nan}.")
        if len(self.x_sample_points) - total_accounted_for - num_still_nan != 0:
            print(f"  Note: some points might not be categorized if near boundaries with slight float variations.")

        self.yvals = sampled_y_values_clipped

        self.sampled_data = list(zip(self.x_sample_points, sampled_y_values_clipped))

circ = circuit.Circuit()


Vin = circuit.VoltageSource(1.0)
R1 = circuit.Resistor(10e3)
C1 = circuit.Capacitor(15e-9)
C2 = circuit.Capacitor(0.56e-9)
R2 = circuit.Resistor(10e3)

for thing in (Vin, R1, C1, C2, R2):
    circ.addComponent(thing)

circ.connectComponents(Vin, 'p', C1, 'p')
circ.connectComponents(Vin, 'n', None, None)  
circ.connectComponents(C1, 'n', R1, 'p')
circ.connectComponents(C1, 'n', R2, 'p')
circ.connectComponents(R2, 'n', C2, 'p', True)
circ.connectComponents(R1, 'n', None, None)
circ.connectComponents(R1, 'n', C2, 'n')

f = np.logspace(1, 6, 400)           
mag = np.zeros_like(f, dtype=float)
phase = np.zeros_like(f, dtype=float)

for i, freq in enumerate(f):
    omega = 2 * np.pi * freq
    circ.setFrequency(omega)
    V = circ.solveSystem()
    Vout = V[circ.VOUT]
    mag[i] = 20 * np.log10(np.abs(Vout))
    phase[i] = np.angle(Vout, deg=True)

interactive_drawer = InteractiveLogDraw(f)
print(len(interactive_drawer.yvals))

mag = interactive_drawer.yvals

######################################################################################################################################################################
######################################################################################################################################################################
######################################Sample Circuits########################################################################################
######################################################################################################################################################################
######################################################################################################################################################################

low_pass = circuit.Circuit()
low_pass_v1 = circuit.VoltageSource(1.0)
R1_lp = circuit.Resistor(10e3)
C1_lp = circuit.Capacitor(10e-9)

for thing in (R1_lp, C1_lp, low_pass_v1):
    low_pass.addComponent(thing)

low_pass.connectComponents(low_pass_v1, 'n', None, None)  
low_pass.connectComponents(low_pass_v1, 'p', R1_lp, 'p')
low_pass.connectComponents(R1_lp, 'n', C1_lp, 'p', True)
low_pass.connectComponents(C1_lp, 'n', None, None)

mag_lp = np.zeros_like(f, dtype=float)
phase_lp = np.zeros_like(f, dtype=float)

for i, freq in enumerate(f):
    omega = 2 * np.pi * freq
    low_pass.setFrequency(omega)
    V = low_pass.solveSystem()
    Vout = V[low_pass.VOUT]
    mag_lp[i] = 20 * np.log10(np.abs(Vout))
    phase_lp[i] = np.angle(Vout, deg=True)

####################

high_pass = circuit.Circuit()
high_pass_v1 = circuit.VoltageSource(1.0)
R1_hp = circuit.Resistor(10e3)
C1_hp = circuit.Capacitor(10e-9)

for thing in (R1_hp, C1_hp, high_pass_v1):
    high_pass.addComponent(thing)

high_pass.connectComponents(high_pass_v1, 'n', None, None)  
high_pass.connectComponents(high_pass_v1, 'p', C1_hp, 'p')
high_pass.connectComponents(C1_hp, 'n', R1_hp, 'p', True)
high_pass.connectComponents(R1_hp, 'n', None, None)

mag_hp = np.zeros_like(f, dtype=float)
phase_hp = np.zeros_like(f, dtype=float)

for i, freq in enumerate(f):
    omega = 2 * np.pi * freq
    high_pass.setFrequency(omega)
    V = high_pass.solveSystem()
    Vout = V[high_pass.VOUT]
    mag_hp[i] = 20 * np.log10(np.abs(Vout))
    phase_hp[i] = np.angle(Vout, deg=True)

##################


sk_bp = circuit.Circuit()
Vin = circuit.VoltageSource(1.0)            
sk_bp.addComponent(Vin)
sk_bp.connectComponents(Vin, 'n',  None, None)  

opHP  = circuit.IdealOpAmp()
C1_BP = circuit.Capacitor(10e-9)         
C2_BP = circuit.Capacitor(10e-9)
R1_BP = circuit.Resistor (10e3)            
R2_BP = circuit.Resistor (200e3)

for c in (opHP, C1_BP, C2_BP, R1_BP, R2_BP):
    sk_bp.addComponent(c)


sk_bp.connectComponents(Vin,   'p', C1_BP, 'p')      
sk_bp.connectComponents(C1_BP, 'n', opHP,  'V+')     
sk_bp.connectComponents(C1_BP, 'n', R1_BP,'p')      
sk_bp.connectComponents(R1_BP, 'n', None,  None)
sk_bp.connectComponents(opHP,  'Vout', C2_BP,'n')   
sk_bp.connectComponents(C2_BP, 'p', C1_BP,'n')       
sk_bp.connectComponents(opHP,  'Vout', opHP,'V-')   

opLP  = circuit.IdealOpAmp()
R1_bp_LP = circuit.Resistor (10e3)
R2_bp_LP = circuit.Resistor (10e3)
C1_bp_LP = circuit.Capacitor(10e-9)
C2_bp_LP = circuit.Capacitor(10e-9)

for c in (opLP, R1_bp_LP, R2_bp_LP, C1_bp_LP, C2_bp_LP):
    sk_bp.addComponent(c)

sk_bp.connectComponents(opHP,'Vout', R1_bp_LP,'p')     
sk_bp.connectComponents(R1_bp_LP,'n',   opLP,'V+')     
sk_bp.connectComponents(R1_bp_LP,'n',   C1_bp_LP,'p')    
sk_bp.connectComponents(C1_bp_LP,'n',   None,  None)

sk_bp.connectComponents(opLP,'Vout', R2_bp_LP,'p')     
sk_bp.connectComponents(R2_bp_LP,'n',   R1_bp_LP,'n')     
sk_bp.connectComponents(opLP,'Vout', C2_bp_LP,'n')     
sk_bp.connectComponents(C2_bp_LP,'p',   None,  None)
sk_bp.connectComponents(opLP,'Vout', opLP,'V-')     


opG   = circuit.IdealOpAmp()
RinG  = circuit.Resistor(10e3)   
RfG   = circuit.Resistor(190e3)   

for c in (opG, RinG, RfG):
    sk_bp.addComponent(c)


sk_bp.connectComponents(opLP, 'Vout', opG,  'V+')


sk_bp.connectComponents(opG,  'Vout', RfG, 'p')
sk_bp.connectComponents(RfG,  'n',    opG, 'V-')
sk_bp.connectComponents(opG,  'V-',   RinG,'p')
sk_bp.connectComponents(RinG, 'n',    None, None)   

VOUT_NODE = opG.Vout      
sk_bp.VOUT = VOUT_NODE
mag_bp   = np.zeros_like(f)
phase_bp = np.zeros_like(f)

for i, freq in enumerate(f):
    sk_bp.setFrequency(2*np.pi*freq)
    V = sk_bp.solveSystem()
    Vout = V[VOUT_NODE]
    mag_bp[i]   = 20*np.log10(abs(Vout))
    phase_bp[i] = np.angle(Vout, deg=True)

###################


bs = circuit.Circuit()
bs_vin = circuit.VoltageSource(1.0)
bs.addComponent(bs_vin)
bs.connectComponents(bs_vin, 'n',  None, None)  

C1 = circuit.Capacitor(20e-9)
C2 = circuit.Capacitor(20e-9)
C3 = circuit.Capacitor(40e-9)
R1 = circuit.Resistor(2500)
R2 = circuit.Resistor(2500)
R3 = circuit.Resistor(5000)

for i in (C1, C2, C3, R1, R2, R3):
    bs.addComponent(i)

bs.connectComponents(bs_vin, 'p', C1, 'p')
bs.connectComponents(bs_vin, 'p', R1, 'p')
bs.connectComponents(C1, 'n', R3, 'p')
bs.connectComponents(R1, 'n', C3, 'p')
bs.connectComponents(R3, 'n', None, None)
bs.connectComponents(C3, 'n', None, None)
bs.connectComponents(R1, 'n', R2, 'p')
bs.connectComponents(C1, 'n', C2, 'p')
bs.connectComponents(C2, 'n', R2, 'n', True)

mag_bs   = np.zeros_like(f)
phase_bs = np.zeros_like(f)

for i, freq in enumerate(f):
    bs.setFrequency(2*np.pi*freq)
    V = bs.solveSystem()
    Vout = V[bs.VOUT]
    mag_bs[i]   = 20*np.log10(abs(Vout))
    phase_bs[i] = np.angle(Vout, deg=True)



#####################

circuit_list = [(low_pass, mag_lp, phase_lp, "low_pass rc"), (high_pass, mag_hp, phase_hp, "high_pass rc"), (sk_bp, mag_bp, phase_bp, "sallen key band pass"), (bs, mag_bs, phase_bs, "band stop")]

plt.figure()
plt.semilogx(f, mag)
plt.semilogx(f, mag_lp)
plt.semilogx(f, mag_hp)
plt.semilogx(f, mag_bp)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Sallen‑Key Low‑Pass Filter – Magnitude Response")
plt.grid(True, which='both')


plt.figure()
#plt.semilogx(f, phase)
plt.semilogx(f, phase_lp)
plt.semilogx(f, phase_hp)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (deg)")
plt.title("Sallen‑Key Low‑Pass Filter – Phase Response")
plt.grid(True, which='both')
plt.show()


#####################

def compute_bode(circ, f, mag, phase):
    for i, freq in enumerate(f):
        omega = 2 * np.pi * freq
        circ.setFrequency(omega)
        V = circ.solveSystem()
        Vout = V[circ.VOUT]
        mag[i] = 20 * np.log10(np.abs(Vout))
        phase[i] = np.angle(Vout, deg=True)
    return mag

# def compute_loss(a, b, c):
#     mag = a[0]
#     phase = a[1]
#     return np.dot(mag-b, mag-b)+np.dot(phase -c, phase - c)

def compute_loss(a, b):
    return np.dot(a-b, a-b)

#####################

desired_mag = np.copy(mag)
total_mag = np.zeros_like(f, dtype=float)
initial_loss = sys.maxsize * 2 + 1
l = 0
while (l < 5):
    losses = []
    mags = []
    for index in range(len(circuit_list)):
        #index = np.argmax([np.dot(desired_mag, base[1])/(np.linalg.norm(desired_mag) * np.linalg.norm(base[1])) for base in circuit_list])
        #index =  np.argmax([compute_loss(desired_mag, base[1]) for base in circuit_list]) 
        #could use these prior lines to guess best circuit, but could also brute force, lwk brute-forcing is kind of useless cuz the guess
        #is lwk almost always right
        #print(circuit_list[index][-1])
        base_circ = circuit_list[index][0]
        nu = 0.00001
        eta = 0.00001
        max_change_factor = 0.1 
        mag_p = circuit_list[index][1]
        phase_p = circuit_list[index][2]
        for k in range(150):
            changes = []
            original_L = compute_loss(compute_bode(base_circ, f, mag_p, phase_p), desired_mag)
            for comp in base_circ.components:
                if isinstance(comp, circuit.Resistor) or isinstance(comp, circuit.Capacitor):
                    original_val = comp.value
                    epsilon = max(nu * original_val, 0.000000000001)
                    comp.value += epsilon
                    perturbed_L =  compute_loss(compute_bode(base_circ, f, mag_p, phase_p), desired_mag)
                    dLdX = (perturbed_L - original_L)/epsilon
                    change = -1 * eta * dLdX
                    comp.value = original_val
                    if abs(change) > abs(max_change_factor * comp.value):
                        change = math.copysign(max_change_factor * comp.value, change)
                    changes.append(change)
                else:
                    changes.append('N/A')

            for i in range(len(base_circ.components)):
                if not changes[i] == 'N/A':
                    base_circ.components[i].value = max(1e-10, base_circ.components[i].value + changes[i])
            
            if (k % 50 == 0):
                mag_p = compute_bode(base_circ, f, mag_p, phase_p)
                print(f"Optimizing Round {k}: {compute_loss(mag_p, desired_mag)}")
        mag_p = compute_bode(base_circ, f, mag_p, phase_p)
        loss_p = compute_loss(desired_mag, mag_p)
        mags.append(mag_p)
        losses.append(loss_p) 
    
    if min(losses) > initial_loss:
        break 
    mag_p = mags[np.argmin(losses)]
    print(circuit_list[np.argmin(losses)][-1])
    initial_loss = min(losses)
    desired_mag = desired_mag - mag_p
    total_mag = total_mag + mag_p
    l += 1
    


plt.figure()
plt.semilogx(f, mag)
plt.semilogx(f, total_mag)
plt.semilogx(f, mag - total_mag)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Sallen‑Key Low‑Pass Filter – Magnitude Response")
plt.grid(True, which='both')
plt.show()
        
    
    
    
