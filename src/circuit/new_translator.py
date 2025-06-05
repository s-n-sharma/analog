# translator.py
import re, copy, pathlib
from collections import defaultdict
from Circuit import (Circuit, SubCircuit,
                     Resistor, Capacitor, VoltageSource, IdealOpAmp)


_METRIC = {'':1, 'f':1e-15, 'p':1e-12, 'n':1e-9,
           'u':1e-6, 'm':1e-3,  'k':1e3, 'meg':1e6, 'g':1e9}

def _num(token: str) -> float:
    m = re.fullmatch(r'([-+]?[\d.]+(?:[eE][-+]?\d+)?)([a-z]*)', token)
    if not m: raise ValueError(f"Bad numeric token {token}")
    base, suff = m.groups()
    return float(base) * _METRIC[suff.lower()]


def _logical_lines(fileobj):
    buf = ''
    for raw in fileobj:
        line = raw.split('*', 1)[0].rstrip()
        if not line: continue
        if line.startswith('+'):
            buf += ' ' + line[1:].lstrip()
            continue
        if buf: yield buf
        buf = line
    if buf: yield buf
    
class Translator:
    """
    * library     : {name : SubCircuit}  – all cached blocks
    * main_circuit: Circuit              – parsed top level
    """
    def __init__(self, path, extra_lib_dirs=()):
        if isinstance(extra_lib_dirs, (str, pathlib.Path)):
            extra_lib_dirs = [extra_lib_dirs]      # <— add this
        self.library = {}
        for d in extra_lib_dirs:
            self._scan_library_dir(d)
        self.main_circuit = self._parse_file(path)

    def _scan_library_dir(self, folder):
        for p in pathlib.Path(folder).rglob("*.sp*"):
            # fully parse the file so a top‑level .declare_subckt becomes cached
            self._parse_file(p)          # ignore the returned Circuit

    def _parse_file(self, path, *, cache_only=False):
        with open(path) as f:
            lines = list(_logical_lines(f))

        # Pass 1: split into blocks
        main_lines, subckt_blocks = self._split_into_blocks(lines)
        print("reached here", path)
        print(main_lines)
        
        # Pass 2: build & cache every sub‑circuit in this file
        for name, blk in subckt_blocks.items():
            if name not in self.library:           # don’t overwrite
                self.library[name] = self._build_subcircuit(name, blk)

        if cache_only: return None                 # used only for library scan

        # Pass 3: parse the top level
        circ, declared_ports = self._parse_lines_into_circuit(main_lines)

        # Pass 4: if the file declares itself as a subckt – cache it
        if declared_ports:
            name = pathlib.Path(path).stem.upper()
            sc   = SubCircuit(Circuit())
            sc.components   = copy.deepcopy(circ.components)
            sc.setIO_nodes(*declared_ports)
            self.library[name] = sc
        print(circ.components)
        return circ

  
    @staticmethod
    def _split_into_blocks(lines):
        main, blocks, curr, curr_name = [], {}, [], None
        for ln in lines:
            m = re.match(r'\.subckt\s+(\w+)', ln, re.I)
            if m:                      
                curr_name = m.group(1).upper()
                curr      = []
                continue
            if re.match(r'\.ends', ln, re.I):       # end block
                blocks[curr_name] = curr[:]
                curr_name = None
                continue
            (curr if curr_name else main).append(ln)
        return main, blocks

    def _build_subcircuit(self, name, lines):
        sc, _ = self._parse_lines_into_circuit(lines, inside_subckt=True)
        # inside_subckt=True guarantees we saw .declare_subckt      return sc


    def _parse_lines_into_circuit(self, lines, *, inside_subckt=False):
        circ      = Circuit()
        ports     = None    
        for ln in lines:
            tok = ln.split()
            if not tok: continue

            if tok[0].lower() == '.declare_subckt':
                if len(tok) != 3:
                    raise ValueError(".declare_subckt needs exactly two pins")
                ports = (self._node_id(tok[1]), self._node_id(tok[2]))
                continue
            if tok[0].upper() == 'X':
                *nodes, name = tok[1:]
                self._instantiate(circ, name.upper(),
                                  [self._node_id(t) for t in nodes])
                continue


            self._add_primitive(circ, tok)

        if inside_subckt and not ports:
            raise ValueError("Sub‑circuit is missing .declare_subckt")
        return circ, ports

    # ------------------------------------------------------------------ primitives
    def _add_primitive(self, circ, tok):
        kind  = tok[0][0].upper()
        n1, n2 = self._node_id(tok[1]), self._node_id(tok[2])

        if kind == 'R':
            comp = Resistor(_num(tok[3])); comp.p, comp.n = n1, n2
        elif kind == 'C':
            comp = Capacitor(_num(tok[3])); comp.p, comp.n = n1, n2
        elif kind == 'V':
            comp = VoltageSource(_num(tok[3])); comp.p, comp.n = n1, n2
        elif tok[0].upper() == 'IOP':
            comp = IdealOpAmp()
            comp.Vplus, comp.Vminus, comp.Vout = n1, n2, self._node_id(tok[3])
            circ.addComponent(comp); return
        else:
            raise ValueError(f"Unknown primitive line: {' '.join(tok)}")

        circ.addComponent(comp)

    # ------------------------------------------------------------------ subckt instantiation
    def _instantiate(self, parent, name, actual_nodes):
        print(name)
        if name not in self.library:
            raise KeyError(f"Unknown subcircuit {name}")
        proto = self.library[name]
        if len(actual_nodes) != 2:
            raise ValueError(f"{name} expects 2 pins")
        inst  = copy.deepcopy(proto)

        # offset all internal nodes to avoid collisions
        max_node = max([0]+[getattr(c, a) or 0
                            for c in parent.components
                            for a in ('p','n','Vplus','Vminus','Vout')])
        offset   = max_node + 1
        for c in inst.components:
            for a in ('p','n','Vplus','Vminus','Vout'):
                if hasattr(c, a) and getattr(c, a):
                    setattr(c, a, getattr(c, a)+offset)

        # map declared ports
        in_pin, out_pin = actual_nodes
        for c in inst.components:
            if isinstance(c, IdealOpAmp):
                if c.Vplus  == inst.inputNode + offset:  c.Vplus  = in_pin
                if c.Vminus == inst.inputNode + offset:  c.Vminus = in_pin
                if c.Vout   == inst.outputNode+ offset:  c.Vout   = out_pin
            else:
                if c.p == inst.inputNode + offset: c.p = in_pin
                if c.n == inst.inputNode + offset: c.n = in_pin
                if c.p == inst.outputNode+ offset: c.p = out_pin
                if c.n == inst.outputNode+ offset: c.n = out_pin

        for c in inst.components:
            parent.addComponent(c)

    # ------------------------------------------------------------------ token → node int
    @staticmethod
    def _node_id(tok):
        if tok == '0': return 0
        if re.fullmatch(r'\d+', tok): return int(tok)
        # letters used inside subckt => allocate negative IDs to avoid clash
        return hash(tok) & 0x7FFF_FFFF