import matplotlib.pyplot as plt
import numpy as np

"""basic class to handle plotting"""
class Plotter:
    def __init__(self, circuit, output_node, freq_range=(1, 1e5), num_points=400):
        """
        Initialize the plotter
        
        Args:
            circuit: Circuit object to analyze
            output_node: Node to measure output voltage from
            freq_range: Tuple of (min_freq, max_freq) in Hz
            num_points: Number of frequency points to compute
        """
        self.circuit = circuit
        self.output_node = output_node
        self.freq_range = freq_range
        self.num_points = num_points
        
        # Generate frequency points
        self.frequencies = np.logspace(np.log10(freq_range[0]), 
                                     np.log10(freq_range[1]), 
                                     num_points)
        
        # Pre-allocate arrays for results
        self.magnitudes = np.zeros(num_points, dtype=float)
        self.phases = np.zeros(num_points, dtype=float)
        
        self.compute_response()
        
    def compute_response(self):
        """Compute the frequency response using vectorized operations"""
        # Convert frequencies to angular frequencies
        omegas = 2 * np.pi * self.frequencies
        
        # Compute response for each frequency
        for i, omega in enumerate(omegas):
            self.circuit.setFrequency(omega)
            V = self.circuit.solveSystem()
            if V is None:
                continue
                
            Vout = V[self.output_node]
            self.magnitudes[i] = 20 * np.log10(np.abs(Vout))
            self.phases[i] = np.angle(Vout, deg=True)
            
        return self.magnitudes, self.phases
    
    def plot_magnitude(self, title=None, show=True):
        """Plot the magnitude response"""
        plt.figure(figsize=(10, 6))
        plt.semilogx(self.frequencies, self.magnitudes)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        if title:
            plt.title(title)
        else:
            plt.title("Magnitude Response")
        plt.grid(True, which='both')
        if show:
            plt.show()
            
    def plot_phase(self, title=None, show=True):
        """Plot the phase response"""
        plt.figure(figsize=(10, 6))
        plt.semilogx(self.frequencies, self.phases)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (degrees)")
        if title:
            plt.title(title)
        else:
            plt.title("Phase Response")
        plt.grid(True, which='both')
        if show:
            plt.show()
            
    def plot_bode(self, title=None, show=True):
        """Plot both magnitude and phase responses"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Magnitude plot
        ax1.semilogx(self.frequencies, self.magnitudes)
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True, which='both')
        
        # Phase plot
        ax2.semilogx(self.frequencies, self.phases)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (degrees)")
        ax2.grid(True, which='both')
        
        if title:
            fig.suptitle(title)
            
        plt.tight_layout()
        if show:
            plt.show()
            
    def save_plots(self, filename_prefix):
        """Save both magnitude and phase plots to files"""
        self.plot_magnitude(show=False)
        plt.savefig(f"{filename_prefix}_magnitude.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plot_phase(show=False)
        plt.savefig(f"{filename_prefix}_phase.png", dpi=300, bbox_inches='tight')
        plt.close()