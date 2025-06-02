"""
This file produces passive circuits and their magnitude and phase plots, saving them to a file in the following format:
Circuit Netlist
--------
Frequency Magnitude
--------
Frequency Phase
"""

from scipy.cluster.hierarchy import DisjointSet as djs
import translator 
import Circuit as circ
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random

def min_deg_func(dic):
    for i in dic.keys():
        if dic[i] < 3:
            return False
    return True

MAX_NODES = 5

counter = 0

for i in range(3, MAX_NODES):
    for _ in range(15):
        djs_conn = djs([j for j in range(i + 3)])
        with open(f'example_circuits/graph_{counter}.txt', 'w') as f:
            EDGE_NUMBER = random.randint(int(1.5*i), int((i+3)*(i+2)*0.25)) #want graph to be pretty connected
            min_deg_2 = False
            deg_count = {k : 0 for k in range(i+3)}
            j = 0
            Resistor_count = 0
            Capacitor_count = 0
            #0 is Vin, i+1 is ground, i+2 is vout
            f.write("V1 Vin 0 1\n")
            while (j < EDGE_NUMBER or not djs_conn.connected(0, i) or not djs_conn.connected(i, i+2) or not djs_conn.connected(i+1, i+2) or not (len(djs_conn.subsets()) == 1) or not min_deg_2):
                node_1 = random.randint(0, i+2)
                node_2 = random.randint(0, i+2)
                deg_count[node_1] += 1
                deg_count[node_2] += 1
                while (node_2 == node_1):
                    node_2 = random.randint(0, i+2)
                djs_conn.merge(node_1, node_2)
                component_type = random.randint(0, 1)
                if (component_type == 0):
                    component = "R"+str(Resistor_count+1)
                    Resistor_count += 1
                    unit = "k"
                else:
                    component = "C"+str(Capacitor_count+1)
                    Capacitor_count += 1
                    unit = "n"
                
                if node_1 == 0:
                    node_1 = "Vin"
                elif node_1 == i+1:
                    node_1 = "0"
                elif node_1 == i+2:
                    node_1 = "Vout"
                else:
                    node_1 = "n"+str(node_1)
                
                if node_2 == 0:
                    node_2 = "Vin"
                elif node_2 == i+1:
                    node_2 = "0"
                elif node_2 == i+2:
                    node_2 = "Vout"
                else:
                    node_2 = "n"+str(node_2)

                comp_strength = random.choice([0.1, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 250, 375, 500, 1000])
                f.write(component+" " + node_1 + " " + node_2+" "+str(comp_strength)+unit+"\n")             
                min_deg_2 = min_deg_func(deg_count)
                j += 1
            counter += 1