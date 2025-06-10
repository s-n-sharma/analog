import re, copy, pathlib
from src.circuit.Circuit import (Circuit,
                     Resistor, Capacitor, VoltageSource, IdealOpAmp)

_METRIC = {'':1, 'f':1e-15, 'p':1e-12, 'n':1e-9,
           'u':1e-6, 'm':1e-3, 'k':1e3, 'meg':1e6, 'g':1e9}

def _to_number(tok: str) -> float:
    m = re.fullmatch(r'([-+]?[\d.]+(?:[eE][-+]?\d+)?)([a-z]*)', tok)
    if not m:
        raise ValueError(f"Bad numeric token {tok}")
    base, suff = m.groups()
    return float(base) *  _METRIC[suff.lower()]

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

class Translator:

    def __init__(self, netlist_path: str, nodeID=0, parentPins=None, subckt_path="./.subckts", function_path="./.functions", parent_node_id=None, param_map=None):
        self.subckt_path   = pathlib.Path(subckt_path)
        self.function_path = pathlib.Path(function_path)
        self.circuit       = Circuit()
        self.token2node    = {'0': nodeID}
        self.next_node_id  = nodeID + 1
        self.nodeID = nodeID
        self.numsubckts = 0
        self.subckt_ports = []
        self.parentPins = parentPins
        self.VOUT = None
        self.parent_node_id = parent_node_id  # Track parent node ID for hierarchical numbering
        self.param_map = param_map or {}  # Parameter mapping for function calls

        self._parse_file(netlist_path)

        if self.VOUT is not None:
            self.circuit.VOUT = self.VOUT

    def _gid(self, tok: str) -> int:     
        if tok == "0":
            return 0  
        
        if tok not in self.token2node:
            # For subcircuits, use parent node ID as base for hierarchical numbering
            if self.parent_node_id is not None:
                new_id = self.parent_node_id * 1000 + self.next_node_id
            else:
                new_id = self.next_node_id
            self.token2node[tok] = new_id
            self.next_node_id += 1
        return self.token2node[tok]

    def _parse_file(self, path: str):       
        for ln in _logical_lines(open(path)):
            tok  = ln.split()
            if not tok:
                continue
            cmd  = tok[0].upper()
            if cmd == '.DECLARE_SUBCKT':
                if len(tok) != 3:
                    raise ValueError(".declare_subckt line must be:  .declare_subckt <in> <out>")
                self.circuit.subckt_nodes = tok[1:]
                self.subckt_ports = tok[1:]
            elif cmd == '.FN':
                if len(tok) < 4:
                    raise ValueError(".fn line must be:  .fn <node1> <node2> <circuit_file> [param=val ...]")
                self._handle_function_call(tok[1:])
                continue

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
                self.circuit.subckt_nodes = [int(self.token2node[tok[1]]), int(self.token2node[tok[2]])]
                continue
            if first_char == 'X':
                *pins_str, sub_name = tok[1:]
                pins_gid           = [self._gid(p) for p in pins_str]

                sub_path = self.subckt_path / f"{sub_name}.sp"
                if not sub_path.exists():
                    raise FileNotFoundError(f"Cannot find sub‑ckt file '{sub_path}'")

                # Pass the current node ID as parent_node_id for hierarchical numbering
                sub_trans = Translator(str(sub_path), 
                                    subckt_path=self.subckt_path, 
                                    nodeID=self.nodeID,
                                    parentPins=pins_gid,
                                    parent_node_id=self.next_node_id - 1)
                self.numsubckts += 1
                subckt = sub_trans.circuit

                if len(pins_gid) != 2:
                    raise ValueError("Exactly two interface pins are expected for a sub‑circuit")

                self.circuit.addSubckt(subckt, pins_gid[0], pins_gid[1])
                continue

    def _handle_function_call(self, tokens):
        """Handle a function call in the netlist."""
        node1, node2, circuit_file = tokens[0:3]
        params = tokens[3:]  # Remaining tokens are parameters
        
        # Get node IDs
        n1, n2 = self._gid(node1), self._gid(node2)
        
        circuit_path = self.function_path / f"{circuit_file}.sp"
        if not circuit_path.exists():
            raise FileNotFoundError(f"Cannot find circuit file '{circuit_path}'")
            
        param_map = {}
        for param in params:
            if '=' not in param:
                raise ValueError(f"Parameter must be in format 'name=value': {param}")
            name, value = param.split('=', 1)
            param_map[name] = value
            
        func_trans = Translator(str(circuit_path),
                              nodeID=self.next_node_id,
                              parentPins=[n1, n2],
                              subckt_path=self.subckt_path,
                              parent_node_id=self.next_node_id - 1,
                              param_map=param_map)
                              
        # Add all components from the function to the main circuit
        for comp in func_trans.circuit.components:
            self.circuit.addComponent(comp)
            
        # Update node numbering
        self.next_node_id = func_trans.next_node_id
        self.token2node.update(func_trans.token2node)

    def _add_primitive(self, tok: list[str]):
        kind = tok[0][0].upper()

        # Map interface nodes
        n1, n2 = tok[1], tok[2]
        if n1 == 'in':
            n1 = self.parentPins[0] if self.parentPins else self._gid(n1)
        elif n1 == 'out':
            n1 = self.parentPins[1] if self.parentPins else self._gid(n1)
        else:
            n1 = self._gid(n1)
            
        if n2 == 'in':
            n2 = self.parentPins[0] if self.parentPins else self._gid(n2)
        elif n2 == 'out':
            n2 = self.parentPins[1] if self.parentPins else self._gid(n2)
        else:
            n2 = self._gid(n2)
    
        if kind == 'R':
            value = tok[3]
            if value in self.param_map:
                value = self.param_map[value]
            comp = Resistor(_to_number(value))
            comp.n, comp.p = n1, n2
        elif kind == 'C':
            value = tok[3]
            if value in self.param_map:
                value = self.param_map[value]
            comp = Capacitor(_to_number(value))
            comp.n, comp.p = n1, n2
        elif kind == 'V':
            value = tok[3]
            if value in self.param_map:
                value = self.param_map[value]
            comp = VoltageSource(_to_number(value))
            comp.n, comp.p = n1, n2
        elif tok[0].upper() == 'IOP':
            comp = IdealOpAmp()
            comp.Vplus, comp.Vminus, comp.Vout = n1, n2, self._gid(tok[3])
            if tok[3] == 'out':
                comp.Vout = self.parentPins[1] if self.parentPins else self._gid(tok[3])
            elif tok[3] == 'in':
                comp.Vout = self.parentPins[0] if self.parentPins else self._gid(tok[3])
            else:
                comp.Vout = self._gid(tok[3])
        else:                                    
            raise ValueError(f"Unhandled primitive line: {' '.join(tok)}")

        self.circuit.addComponent(comp)