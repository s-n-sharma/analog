import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from PyLTSpice.LTSpice_RawRead import LTSpiceRawRead # For parsing .raw files

# --- Configuration ---
# IMPORTANT: Update this path to your LTSpice XVII executable
# Windows example:
# ltspice_executable = "C:\\Program Files\\LTC\\LTspiceXVII\\XVIIx64.exe"
# macOS example (ensure LTspice.app is in Applications):
ltspice_executable = "/Applications/LTspice.app/Contents/MacOS/LTspice" # Adjusted for macOS
# Linux example (using Wine, path might vary):
# ltspice_executable = "wine ~/.wine/drive_c/Program\\ Files/LTC/LTspiceXVII/XVIIx64.exe"

# Circuit parameters
r_value = 1000  # Resistance in Ohms (e.g., 1kOhm)
c_value = 1e-6  # Capacitance in Farads (e.g., 1uF)
output_node_name = "Vout" # Name of the output node in the netlist

# Simulation parameters
start_freq = 1       # Start frequency in Hz
stop_freq = 1e6      # Stop frequency in Hz (1 MHz)
points_per_decade = 100

# File names
netlist_filename = "rc_lowpass_auto.cir"
# raw_filename will be derived from netlist_filename
log_filename = "rc_lowpass_auto.log" # Will also be derived

# --- 1. Generate LTSpice Netlist (.cir file) ---
def generate_netlist(filename, r_val, c_val, v_out_name, start_f, stop_f, n_points):
    # Removed "nonewinsteps=1" and "nopad=1" from .options line
    netlist_content = f"""
* Automated RC Low Pass Filter
V1 Vin 0 AC 1 Rser=0 ; AC voltage source, 1V amplitude, 0 series resistance
R1 Vin {v_out_name} {r_val}
C1 {v_out_name} 0 {c_val}

* AC Analysis
.ac dec {n_points} {start_f} {stop_f}

* Options for batch mode and data saving
.options meascplx=1 numdgt=7 ; Ensure enough precision and complex data for AC
.save V({v_out_name.lower()}) ; Explicitly save the output node voltage

.end
"""
    try:
        with open(filename, "w") as f:
            f.write(netlist_content)
        print(f"Netlist '{filename}' generated successfully.")
    except IOError as e:
        print(f"Error writing netlist file '{filename}': {e}")
        return False
    return True

# --- 2. Run LTSpice Simulation ---
def run_ltspice_simulation(ltspice_exe_path, netlist_path):
    # Auto-detection logic for LTSpice path (mostly for convenience)
    if not os.path.exists(ltspice_exe_path):
        common_paths_mac = ["/Applications/LTspice.app/Contents/MacOS/LTspice"]
        common_paths_win = [
            "C:\\Program Files\\LTC\\LTspiceXVII\\XVIIx64.exe",
            "C:\\Program Files (x86)\\LTC\\LTspiceXVII\\XVIIx64.exe",
        ]
        found_path_updated = None # Store the potentially updated path

        current_os = os.name
        if current_os == 'posix' and os.uname().sysname == 'Darwin': # macOS
            for path in common_paths_mac:
                if os.path.exists(path):
                    found_path_updated = path
                    print(f"Found LTSpice at: {found_path_updated}")
                    break
        elif current_os == 'nt': # Windows
            for path in common_paths_win:
                if os.path.exists(path):
                    found_path_updated = path
                    print(f"Found LTSpice at: {found_path_updated}")
                    break
        
        if found_path_updated:
            ltspice_exe_path = found_path_updated # Use the found path
        else:
            print(f"Error: LTSpice executable not found at the specified path: '{ltspice_exe_path}' and common locations.")
            print("Please check the 'ltspice_executable' variable in the script.")
            return False

    command = [ltspice_exe_path, "-b", netlist_path] # -b for batch mode
    print(f"Running LTSpice command: {' '.join(command)}")

    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        print("LTSpice simulation completed successfully.")
        if process.stdout:
            print("LTSpice STDOUT:\n", process.stdout)
        if process.stderr:
            print("LTSpice STDERR (might contain warnings or info):\n", process.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"LTSpice simulation failed with return code {e.returncode}.")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        return False
    except FileNotFoundError:
        print(f"Error: LTSpice executable not found at '{ltspice_exe_path}'.")
        print("Ensure the path is correct and LTSpice is installed.")
        return False
    except subprocess.TimeoutExpired:
        print("LTSpice simulation timed out.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while running LTSpice: {e}")
        return False

# --- 3. Parse LTSpice Output (.raw file) ---
def parse_raw_file(raw_file_path, v_out_name_netlist):
    if not os.path.exists(raw_file_path):
        print(f"Error: RAW file '{raw_file_path}' not found. Cannot parse.")
        return None, None, None

    try:
        print(f"Parsing RAW file: {raw_file_path}")
        ldr = LTSpiceRawRead(raw_file_path)

        frequency_trace = ldr.get_trace('frequency')
        frequency = frequency_trace.get_wave()

        vout_trace_name_in_raw = f"V({v_out_name_netlist.lower()})"
        available_traces = ldr.get_trace_names()

        if vout_trace_name_in_raw not in available_traces:
            print(f"Warning: Trace '{vout_trace_name_in_raw}' not found directly. Available traces: {available_traces}")
            voltage_traces = [tr for tr in available_traces if tr.startswith('v(') and tr != 'frequency']
            if not voltage_traces:
                 voltage_traces = [tr for tr in available_traces if tr.startswith('V(') and tr != 'frequency']

            if len(voltage_traces) == 1:
                vout_trace_name_in_raw = voltage_traces[0]
                print(f"Using the only available voltage trace: {vout_trace_name_in_raw}")
            elif len(voltage_traces) > 1:
                print(f"Error: Multiple voltage traces found: {voltage_traces}. Cannot automatically determine the correct output.")
                print(f"Please ensure your .save V({v_out_name_netlist}) line in the netlist is specific or adjust 'output_node_name'.")
                return None, None, None
            else:
                print(f"Error: No suitable output voltage trace found. Check .save directive in netlist and .raw file contents.")
                return None, None, None
        
        vout_complex = ldr.get_trace(vout_trace_name_in_raw).get_wave()

        magnitude_db = 20 * np.log10(np.abs(vout_complex))
        phase_degrees = np.angle(vout_complex, deg=True)

        print("RAW file parsed successfully.")
        return frequency, magnitude_db, phase_degrees

    except Exception as e:
        print(f"Error parsing RAW file with PyLTSpice: {e}")
        return None, None, None

# --- 4. Plot Bode Plots ---
def plot_bode(frequency, magnitude_db, phase_degrees, r_val, c_val):
    if frequency is None or magnitude_db is None or phase_degrees is None:
        print("Cannot plot: Data is missing.")
        return

    plt.figure(figsize=(10, 7))

    plt.subplot(2, 1, 1)
    plt.semilogx(frequency, magnitude_db, color='blue')
    plt.title(f'Bode Plot - RC Low Pass Filter (R={r_val}$\Omega$, C={c_val*1e6:.2f}$\mu$F)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which="both", ls='-', alpha=0.7)
    plt.minorticks_on()

    plt.subplot(2, 1, 2)
    plt.semilogx(frequency, phase_degrees, color='red')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (degrees)')
    plt.grid(True, which="both", ls='-', alpha=0.7)
    plt.minorticks_on()

    plt.tight_layout()
    plt.show()

# --- 5. Cleanup Generated Files (Optional) ---
def cleanup_files(*filenames):
    print("\nCleaning up generated files...")
    for f_name in filenames:
        try:
            if os.path.exists(f_name):
                os.remove(f_name)
                print(f"Removed: {f_name}")
        except OSError as e:
            print(f"Error removing file {f_name}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    if not generate_netlist(netlist_filename, r_value, c_value, output_node_name, start_freq, stop_freq, points_per_decade):
        exit(1)

    if not run_ltspice_simulation(ltspice_executable, netlist_filename):
        actual_log_filename = os.path.splitext(netlist_filename)[0] + ".log"
        if os.path.exists(actual_log_filename):
            print(f"\n--- Contents of {actual_log_filename} (check for simulation errors) ---")
            try:
                with open(actual_log_filename, 'r') as logf:
                    print(logf.read())
            except Exception as e:
                print(f"Could not read log file: {e}")
            print(f"--- End of {actual_log_filename} ---")
        exit(1)

    actual_raw_filename = os.path.splitext(netlist_filename)[0] + ".raw"
    frequency, magnitude_db, phase_degrees = parse_raw_file(actual_raw_filename, output_node_name)

    if frequency is not None and magnitude_db is not None and phase_degrees is not None:
        plot_bode(frequency, magnitude_db, phase_degrees, r_value, c_value)
    else:
        print("Plotting skipped due to data parsing errors or missing data.")
        actual_log_filename = os.path.splitext(netlist_filename)[0] + ".log"
        if os.path.exists(actual_log_filename):
            print(f"\n--- Contents of {actual_log_filename} (check for simulation or data saving errors) ---")
            try:
                with open(actual_log_filename, 'r') as logf:
                    print(logf.read())
            except Exception as e:
                print(f"Could not read log file: {e}")
            print(f"--- End of {actual_log_filename} ---")
        exit(1)

    # print("\nProcess finished. To clean up files, uncomment the cleanup_files line in the script.")
    # actual_log_filename = os.path.splitext(netlist_filename)[0] + ".log"
    # cleanup_files(netlist_filename, actual_raw_filename, actual_log_filename)