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
    """
    Read a (very small‑signal) SPICE‑style net‑list and return
    {component‑object: [node‑names]} ready for `build_circuit`.

    Conventions handled
    -------------------
    • Two‑terminal devices   :  line    'R1  n‑  n+  value'
                                stored  ['n+', 'n‑']
    • IOP op‑amp instance    :  line    'IOP U1  n+  n‑  nout'
                                stored  ['nout', 'n+', 'n‑']
    • .subckt / .ends blocks :  parsed but *ignored* (black‑box macro)
    """
    comps = {}
    vout_alias  = None
    with open(path) as f:
        inside_subckt = False
        for line in cleaned_lines(f):
            tok = line.split()
            if not tok:
                continue

            # ─── Skip / leave sub‑ckt definitions ────────────────────
            hd = tok[0].lower()

            # ── skip labels but *keep* the body ──────────────────────
            if hd == '.subckt':              # header: ignore, then parse body
                continue
            if hd in ('.ends', '.end'):      # footer: ignore
                continue


            name = tok[0]
            hd   = name.upper()

            # ─── IOP op‑amp instance (no value token) ────────────────
            if hd.startswith('IOP'):
                nodes = tok[1:]                       # V+  V‑  Vout
                if len(nodes) != 3:
                    raise ValueError("IOP needs exactly 3 nodes")
                nodes = [nodes[2], nodes[0], nodes[1]]  # → [Vout, V+, V‑]
                comp  = DEVICE_FACTORY['IOP'](name, None)
                comps[comp] = nodes
                continue
            
            if hd in ('VOUT', '.VOUT') and len(tok) >= 2:
                vout_alias = tok[1]
                continue            # nothing more on this line
            # ─── Two‑terminal devices  R / C / V … ───────────────────
            nodes = tok[1:-1]     # [neg, pos] in your convention
            if len(nodes) != 2:
                raise ValueError(f"{name}: need exactly two nodes")
            nodes = [nodes[1], nodes[0]]              # → [pos, neg]
            value = tok[-1]

            key = name[0].upper()                     # first letter
            if key not in DEVICE_FACTORY:
                raise ValueError(f"Unknown device type {name}")
            comp = DEVICE_FACTORY[key](name, value)
            comps[comp] = nodes
    if vout_alias is not None:
        for nodes in comps.values():
            for i, n in enumerate(nodes):
                if n == vout_alias:
                    nodes[i] = 'VOUT'
    return comps
def build_circuit(components):
    circuit = circ.Circuit()
    pin_out = {}

    for comp, nodes in components.items():
        circuit.addComponent(comp)
        if isinstance(comp, circ.IdealOpAmp):          
            vout, vp, vn = nodes
            pin_out.setdefault(vout, []).append(('vout', comp))
            pin_out.setdefault(vp,   []).append(('v+',   comp))
            pin_out.setdefault(vn,   []).append(('v-',   comp))
        else:                                       
            p, n = nodes
            pin_out.setdefault(p, []).append(('p', comp))
            pin_out.setdefault(n, []).append(('n', comp))


    for node, conns in pin_out.items():
        base_pin = conns[0]
        for other in conns[1:]:
            if node.upper() == "VOUT":                
                circuit.connectComponents(base_pin[1], base_pin[0],
                                            other[1],  other[0], isVOUT=1)
            circuit.connectComponents(base_pin[1], base_pin[0],
                                        other[1],  other[0])
            
    if circuit.VOUT is None and 'VOUT' in pin_out:
        pin, comp = pin_out['VOUT'][0]             # single (pin‑name, obj)

        # find/create a node number
        node_id = None
        if isinstance(comp, circ.IdealOpAmp):
            node_id = {'vout':  'Vout',
                       'v+':    'Vplus',
                       'v-':    'Vminus'}[pin]
            node_id = getattr(comp, node_id)       # may still be None
        else:
            node_id = getattr(comp, 'p' if pin == 'p' else 'n')

        if node_id is None:                        # not stamped yet
            node_id = circuit.next_node_id
            circuit.next_node_id += 1
            # write it back to the component
            if isinstance(comp, circ.IdealOpAmp):
                setattr(comp, {'vout':'Vout','v+':'Vplus','v-':'Vminus'}[pin],
                        node_id)
            else:
                if pin == 'p': comp.p = node_id
                else:          comp.n = node_id

        circuit.VOUT = node_id                     # finally expose it

    return circuit

