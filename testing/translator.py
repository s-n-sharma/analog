import numpy as np
import Circuit as circ
import re
from collections import namedtuple
from itertools import chain

_METRIC = {'':1, 'f':1e-15, 'p':1e-12, 'n':1e-9, 'u':1e-6,
           'm':1e-3, 'k':1e3, 'meg':1e6, 'g':1e9}

DEVICE_FACTORY = {
    'R': lambda name, val: circ.Resistor(num(val)),
    'C': lambda name, val,**k: circ.Capacitor(num(val)),
    'V': lambda name, val: circ.VoltageSource(num(val)),  
    'IOP': lambda name, val: circ.IdealOpAmp(),      # subckt instance (handled later)
}
def num(s: str) -> float:
    m = re.fullmatch(r'([-+]?[\d.]+(?:[eE][-+]?\d+)?)([a-zA-Z]*)', s)
    if not m:
        raise ValueError(f"Bad numeric token {s}")
    base, suff = m.groups()
    return float(base) * _METRIC[suff.lower()]

def cleaned_lines(fileobj):
    """Yield logical lines with continuations merged,
       comments/blank lines stripped."""
    buf = ''
    for raw in fileobj:
        # Remove text after unescaped '*'  (LTspice style)
        line = raw.split('*', 1)[0].rstrip()
        if not line:
            continue

        if line.startswith('+'):
            buf += ' ' + line[1:].lstrip()
            continue

        if buf:
            yield buf
        buf = line
    if buf:
        yield buf

def parse_netlist(path: str):
    components = {} # stores edges and incident nodes
    with open(path) as f:
        for line in cleaned_lines(f):
            token = line.split()
            name, nodes, value = token[0], token[1: -1], token[-1]
            print(name, nodes, value)
            try:
                comp = DEVICE_FACTORY[name[0]](name, value)
            except:
                comp = DEVICE_FACTORY[name[:3]](name, value)
            components.update({comp: nodes})
    
    return components

def build_circuit(components):
    circuit = circ.Circuit()
    pin_out = {}
    for comp in components:
        circuit.addComponent(comp)
        nodes = components[comp]
        if isinstance(comp, circ.IdealOpAmp):
            vout, vp, vn = nodes
            if not pin_out.__contains__(vout):
                pin_out[vout] = []
            pin_out[vout].append(('vout', comp))
            if not pin_out.__contains__(vp):
                pin_out[vp] = []
            pin_out[vp].append(('v+', comp))
            if not pin_out.__contains__(vn):
                pin_out[vn] = []
            pin_out[vn].append(('v-', comp))

        else:
            p,n = nodes
            if not pin_out.__contains__(p):
                pin_out[p] = []
            pin_out[p].append(('p', comp))
            if not pin_out.__contains__(n):
                pin_out[n] = []
            pin_out[n].append(('n', comp))
    
    for node in pin_out:
        connectedComps = pin_out[node]
        start = connectedComps[0]
        for comp in connectedComps[1:]:
            circuit.connectComponents(start[1], start[0], comp[1], comp[0])
    
    return circuit
