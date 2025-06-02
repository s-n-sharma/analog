import matplotlib.pyplot as plt
import numpy as np

"""basic class to handle plotting"""
class Plotter:
    def __init__(self,frequency, magnitude, phase, name=None):
        self.frequency = frequency
        self.magnitude = magnitude
        self.phase = phase
        self.name = name
        
    def plotMagnitude(self):
        
        plt.figure()
        plt.semilogx(self.frequency, self.magnitude)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        if self.name:
            plt.title(f"{self.name} – Magnitude Response")
        else:
            plt.title(f"Magnitude Response plot")
        plt.grid(True, which='both')
        plt.show()
    def plotPhase(self):
        plt.figure()
        plt.semilogx(self.frequency, self.magnitude)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (degrees)")
        if self.name:
            plt.title(f"{self.name} – Phase Response")
        else:
            plt.title(f"Phase Response plot")
        plt.grid(True, which='both')
        plt.show()