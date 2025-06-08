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
import translator
import sys
import os
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


f = np.logspace(1, 6, 400)           

interactive_drawer = InteractiveLogDraw(f)
#print(len(interactive_drawer.yvals))

mag = interactive_drawer.yvals

def compute_randomness(arr, size):
    ret = []
    for i in range(int(len(arr)/size)):
        avg = sum(arr[5*i:5*(i+1)])/size
        ret.extend([arr[5*i] - avg, arr[5*i + 1] - avg, arr[5*i + 2] - avg, arr[5*i + 3] - avg, arr[5*i + 4] - avg])
    return np.dot(ret, ret)/np.dot(arr, arr)

DIR = 'ex_circ_mag'

count = 0
all_circ = []
file_dic = {}
mag_dic = {}

for filename in os.listdir(DIR):
    if not os.path.isfile(os.path.join(DIR, filename)):
        continue
    file = os.path.join(DIR, filename)
    data = []
    hash_data = []
    with open (file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            asdf = float(line.split()[-1].split('\\')[0])
            data.append(float(line.split()[-1].split('\\')[0]))
            hash_data.append(float(line.split()[-1].split('\\')[0]))
            if np.isnan(asdf) or math.isinf(asdf):
                continue
            print(asdf)
            
    data = np.array(data)
    #all_circ.append(data)
    if not compute_randomness(data, 5) > 0.03:
        count += 1
        all_circ.append(data)
        print(len(all_circ))
        file_dic[str(hash_data)] = filename
        file_dic[str(data)] = filename
        mag_dic[filename] = hash_data
        #print(len(data))

#print(len(all_circ), len(all_circ[0]))

all_circ = np.array(all_circ)

k = 25

f = np.logspace(1, 6, 400)           


all_circ = np.array(all_circ)

standardized_data = (all_circ - all_circ.mean(axis = 0)) / all_circ.std(axis = 0)
covariance_matrix = np.cov(standardized_data, ddof = 1, rowvar = False)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
order_of_importance = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[order_of_importance]
sorted_eigenvectors = eigenvectors[:,order_of_importance]
#reduced_data = np.transpose(np.matmul(standardized_data, sorted_eigenvectors[:,:k]))

example_circs = []
example_circs_mags = []

for i in range(k):
    eigen_vec = sorted_eigenvectors[i] + all_circ.mean(axis=0) * np.ones(len(sorted_eigenvectors[i]))
    index = np.argmax([np.dot(all_circ[j], eigen_vec)/(np.dot(all_circ[j], all_circ[j]) * np.dot(eigen_vec, eigen_vec)) for j in range(len(all_circ))])
    example_circs.append(file_dic[str(all_circ[index])])
    example_circs_mags.append(all_circ[index])


def compute_bode(circ, f, mag, phase):
    for i, freq in enumerate(f):
        omega = 2 * np.pi * freq
        circ.setFrequency(omega)
        V = circ.solveSystem()
        Vout = V[circ.VOUT]
        mag[i] = 20 * np.log10(np.abs(Vout))
        phase[i] = np.angle(Vout, deg=True)
    return mag

def compute_loss(a, b):
    return np.dot(a-b, a-b)

#####################

f = np.logspace(1, 6, 400)           



desired_mag = np.copy(mag)
total_mag = np.zeros_like(f, dtype=float)
initial_loss = sys.maxsize * 2 + 1
l = 0
while (l < 5):
    mags = []
    index = np.argmax([np.dot(desired_mag, example_circs_mags[k])/(np.linalg.norm(desired_mag) * np.linalg.norm(example_circs_mags[k]))] for k in range(len(example_circs_mags)))
    base_circ_file = example_circs[index]
    base_circ_trans = translator.Translator('example_circuits/'+base_circ_file)
    base_circ = base_circ_trans.circuit
    nu = 0.00001
    eta = 0.00001
    max_change_factor = 0.1 
    mag_p = example_circs_mags[index]
    #phase_p = circuit_list[index][2]
    phase_p = mag_p
    for k in range(100):
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
    
    if loss_p > initial_loss:
        break 
    initial_loss = min(initial_loss, loss_p)
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
        
    
    
    
