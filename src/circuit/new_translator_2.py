
import re, copy, pathlib
from Circuit import (Circuit,
                     Resistor, Capacitor, VoltageSource, IdealOpAmp)

# ──────────────────────────────────── helpers ────────────────────────────────────
_METRIC = {'':1, 'f':1e-15, 'p':1e-12, 'n':1e-9,
           'u':1e-6, 'm':1e-3, 'k':1e3, 'meg':1e6, 'g':1e9}

def _to_number(tok: str) -> float:
    m = re.fullmatch(r'([-+]?[\d.]+(?:[eE][-+]?\d+)?)([a-z]*)', tok)
    if not m:
        raise ValueError(f"Bad numeric token {tok}")
    base, suff = m.groups()
    return float(base) * _METRIC[suff.lower()]

def _logical_lines(fileobj):
    """
    1. Strip comments (everything after '*')
    2. Merge continuation lines that start with '+'
    """
    buf = ''
    for raw in fileobj:
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

# ────────────────────────────────── Translator ──────────────────────────────────
class Translator:

    def __init__(self, netlist_path: str, *, subckt_path="./.subckts"):
        self.subckt_path   = pathlib.Path(subckt_path)
        self.circuit       = Circuit()
        self.token2node    = {'0': 0}
        self.next_node_id  = 1
        self.VOUT          = None

        self._parse_file(netlist_path)

        if self.VOUT is not None:
            self.circuit.VOUT = self.VOUT


    def _gid(self, tok: str) -> int:
        if tok not in self.token2node:
            self.token2node[tok] = self.next_node_id
            self.next_node_id += 1
        return self.token2node[tok]


    def _parse_file(self, path: str):
        for ln in _logical_lines(open(path)):
            tok  = ln.split()
            if not tok:
                continue
            cmd  = tok[0].upper()

    
            if cmd == '.END':
                return

            if cmd == 'VOUT':
                if len(tok) != 2:
                    raise ValueError("VOUT line must be:  VOUT <node>")
                self.VOUT = self._gid(tok[1])
                continue

            first_char = tok[0][0].upper()
            if first_char in {'R', 'C', 'V'} or cmd == 'IOP':
                self._add_primitive(tok)
                continue
            if cmd == '.DECLARE_SUBCKT':
                if len(tok) != 3:
                    raise ValueError(".declare_subckt line must be:  .declare_subckt <in> <out>")
                self.circuit.subckt_nodes = [int(tok[1]), int(tok[2])]
                continue
            if first_char == 'X':
                *pins_str, sub_name = tok[1:]
                pins_gid           = [self._gid(p) for p in pins_str]

                sub_path = self.subckt_path / f"{sub_name}.sp"
                if not sub_path.exists():
                    raise FileNotFoundError(f"Cannot find sub‑ckt file '{sub_path}'")

                sub_trans = Translator(str(sub_path), subckt_path=self.subckt_path)
                subckt    = sub_trans.circuit

                if len(pins_gid) != 2:
                    raise ValueError("Exactly two interface pins are expected for a sub‑circuit")

                self.circuit.addSubckt(subckt, pins_gid[0], pins_gid[1])
                continue

            raise ValueError(f"Unknown statement: {ln}")

    def _add_primitive(self, tok: list[str]):
        kind = tok[0][0].upper()
        n1, n2 = self._gid(tok[1]), self._gid(tok[2])

        if kind == 'R':
            comp = Resistor(_to_number(tok[3]));      comp.n, comp.p = n1, n2
        elif kind == 'C':
            comp = Capacitor(_to_number(tok[3]));     comp.n, comp.p = n1, n2
        elif kind == 'V':
            comp = VoltageSource(_to_number(tok[3])); comp.n, comp.p = n1, n2
        elif tok[0].upper() == 'IOP':
            comp = IdealOpAmp()
            comp.Vplus, comp.Vminus, comp.Vout = n1, n2, self._gid(tok[3])
        else:                                    
            raise ValueError(f"Unhandled primitive line: {' '.join(tok)}")

        self.circuit.addComponent(comp)