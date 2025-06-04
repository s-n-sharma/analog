import numpy as np

class Resistor:
    """Two-terminal resistor component"""
    def __init__(self, resistance):
        self.value = resistance
        self.p = None  
        self.n = None  
class Capacitor:
    def __init__(self, capacitance):
        self.value = capacitance
        self.p = None  
        self.n = None 
class VoltageSource:
    """Independent voltage source with specified DC voltage"""
    def __init__(self, voltage):
        self.value = voltage
        self.p = None  
        self.n = None  
class IdealOpAmp:
    """Ideal op amp with three terminals: V+, V-, and Vout"""
    def __init__(self):
        self.Vplus = None   
        self.Vminus = None  
        self.Vout = None    

class Circuit:
    """Circuit container that uses Modified Nodal Analysis (MNA) to solve for node voltages"""
    def __init__(self):
        self.components = []
        self.next_node_id = 1  # incremental node ID allocator (0 is reserved for ground)
        self.frequency = 0
        self.VOUT = None
    
    def setFrequency(self, omega):
        self.frequency = omega

    def addComponent(self, comp):
        self.components.append(comp)

    def connectComponents(self, comp1, term1, comp2, term2, isVOUT=False):
        """Connect a terminal of comp1 to a terminal of comp2 (or to a node/ground)."""
        # Helper 
        def get_node(comp, term):
            if comp is None or isinstance(comp, (int, float)):
                # connecting to a node (int) or ground (None/0)
                return 0 if comp is None else int(comp)
            if isinstance(comp, IdealOpAmp):
                if term in ('V+', 'Vp', 'v+', 'positive'):
                    return comp.Vplus
                elif term in ('V-', 'Vminus', 'v-', 'negative'):
                    return comp.Vminus
                elif term.lower() in ('vout', 'out'):
                    return comp.Vout
                else:
                    raise ValueError(f"Unknown op amp terminal '{term}'")
            else:
                # Two-terminal components (Resistor, VoltageSource, etc.)
                if term in ('p', 'pos', 'positive') or term in (1, '1'):
                    return comp.p
                elif term in ('n', 'neg', 'negative') or term in (2, '2'):
                    return comp.n
                else:
                    raise ValueError(f"Unknown component terminal '{term}'")

        # Helper to set a component terminal to a given node ID
        def set_node(comp, term, node_id):
            if isinstance(comp, IdealOpAmp):
                if term in ('V+', 'Vp', 'v+', 'positive'):
                    comp.Vplus = node_id
                elif term in ('V-', 'Vminus', 'v-', 'negative'):
                    comp.Vminus = node_id
                elif term.lower() in ('vout', 'out'):
                    comp.Vout = node_id
            else:
                if term in ('p', 'pos', 'positive') or term in (1, '1'):
                    comp.p = node_id
                elif term in ('n', 'neg', 'negative') or term in (2, '2'):
                    comp.n = node_id

        # Determine node IDs 
        node1 = get_node(comp1, term1)
        node2 = get_node(comp2, term2)
        # If neither side has a node, assign a new node ID
        if node1 is None and node2 is None:
            node_new = self.next_node_id
            self.next_node_id += 1
            node1 = node_new
            node2 = node_new
        elif node1 is None:
            node1 = node2
        elif node2 is None:
            node2 = node1
        # else merge
        elif node1 != node2:
        
            replace_id = node2
            keep_id = node1
            for comp in self.components:
                if isinstance(comp, IdealOpAmp):
                    if comp.Vplus == replace_id: comp.Vplus = keep_id
                    if comp.Vminus == replace_id: comp.Vminus = keep_id
                    if comp.Vout == replace_id: comp.Vout = keep_id
                else:
                    if comp.p == replace_id: comp.p = keep_id
                    if comp.n == replace_id: comp.n = keep_id
            node2 = node1 = keep_id

        # Set the determined node IDs on the component terminals
        if comp1 is not None and not isinstance(comp1, (int, float)):
            set_node(comp1, term1, node1)
        if comp2 is not None and not isinstance(comp2, (int, float)):
            set_node(comp2, term2, node2)
        
        if isVOUT:
            if isVOUT == 1:
                self.VOUT = node1
                return node1
            else:
                self.VOUT = node2
                return node2
    def solveSystem(self):
        """This solves MNA with op amps, literally copied from https://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA5.html"""
        
        
        # The first step is isolate the nodes - this allows us to track the voltages 
        node_ids = set()
        for c in self.components:
            if isinstance(c, IdealOpAmp):
                node_ids.update([n for n in (c.Vplus, c.Vminus, c.Vout) if n])
            else:
                node_ids.update([n for n in (c.p, c.n) if n])
        node_list  = sorted(node_ids)         
        node_index = {nid: i for i, nid in enumerate(node_list)}
        N          = len(node_list)           

        
        # Now we proceed like MNA
        
        # The G matrix is like the KCL matrix - think of the node voltage analysis equations (no voltage sources yet)
        G = np.zeros((N, N), dtype=complex)
        
        # RHS of G matrix
        I = np.zeros(N,      dtype=complex)

        # helper to fill out G
        def _stamp(a, b, y):
            if a: G[node_index[a], node_index[a]] += y
            if b: G[node_index[b], node_index[b]] += y
            if a and b:
                G[node_index[a], node_index[b]] -= y
                G[node_index[b], node_index[a]] -= y

        # fill in using component's nodes - remember all references to a nodes connect back to previously ID'd ones
        for c in self.components:
            if isinstance(c, Resistor):
                _stamp(c.p or 0, c.n or 0, 1.0 / c.value if c.value else 1e9)
            elif isinstance(c, Capacitor):
                _stamp(c.p or 0, c.n or 0, 1j * self.frequency * c.value)

        # MNA means that voltage sources have an extra currnet current source variable
        Vsrc   = [c for c in self.components if isinstance(c, VoltageSource)] # id voltage sources
        M      = len(Vsrc) 
        OpAmp  = [c for c in self.components if isinstance(c, IdealOpAmp)] # op amps are considered voltage sources (out of vout)
        P      = len(OpAmp)

        
        # These matrices are handling some extra constraints for KCL equations - enforcing the addition/subtraction of voltage sources at nodes
        B = np.zeros((N, M + P), dtype=complex)           
        E = np.zeros(M + P,       dtype=complex)          

        # voltage sources
        for j, s in enumerate(Vsrc): # voltage difference across nodes is s.value
            if s.p: B[node_index[s.p], j] += 1
            if s.n: B[node_index[s.n], j] -= 1
            E[j] = s.value

        # op amps - note this like a floating voltage source (for this equation)
        for k, op in enumerate(OpAmp, start=M):
            if op.Vout and op.Vout in node_index:
                B[node_index[op.Vout], k] = 1          

        # these are the current equations - one for every constraint
        C = np.zeros((M + P, N), dtype=complex)

        # KVL rows for the independent V‑sources
        for j, s in enumerate(Vsrc):
            if s.p: C[j, node_index[s.p]] =  1
            if s.n: C[j, node_index[s.n]] = -1

        # golden rule
        for p, op in enumerate(OpAmp, start=M):
            if op.Vplus and op.Vplus in node_index:
                C[p, node_index[op.Vplus]] += 1
            if op.Vminus and op.Vminus in node_index:
                C[p, node_index[op.Vminus]] -= 1
        

        # assemble this 
        A_top    = np.hstack((G, B))
        A_bottom = np.hstack((C, np.zeros((C.shape[0], B.shape[1]))))
        A        = np.vstack((A_top, A_bottom))
        b        = np.hstack((I, E))

        try:
            x = np.linalg.solve(A, b)

            volt = {0: 0.0}
            for nid, idx in node_index.items():
                volt[nid] = x[idx]

            return volt
        except:
            print(self.frequency)
                     