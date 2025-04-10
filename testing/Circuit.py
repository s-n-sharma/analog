import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self):
        self.voltage = 0
        self.connectedComponents = []
        

class Component:
    def __init__(self, value):
        if (value.real != 0):
            Exception("component values are real")
        self.name = "None"
        self.value = value
        self.nodeForward = Node()
        self.nodeForward.connectedComponents.append(self)
        self.nodeBackward = Node()
        self.nodeBackward.connectedComponents.append(self)
        self.current = 0  
        self.impedance = 0
        
        


class Resistor(Component):
    def __init__(self, resistance):
        super().__init__(resistance)
        
    def setFrequency(self, omega):
        self.impedance = self.value

class Capacitor(Component):
    def __init__(self, capacitance):
        super().__init__(capacitance)
    def setFrequency(self, omega):
        self.impedance = 1 / (1j * omega * self.value)
class VoltageSource(Component):
    def __init__(self, value):
        super().__init__(value)
    def setFrequency(self, omega):
        self.impedance = None

class GroundNode(Node):
    def __init__(self):
        super().__init__()

class DependentVoltageSource(VoltageSource):
    # gain and component
    # always doing nodeForward- nodeBackward
    def __init__(self, value, comp):  
        super().__init__(value)
        self.comp = comp
    
class OperationalAmplifier(Component):
    def __init__(self, value, R_in, R_out): # gain value
        super().__init__(value)
        self.name = "Op Amp"
        R_in = Resistor(R_in)
        R_out = Resistor(R_out)
        ground = GroundNode()
        
        self.nodeForward = R_out.nodeBackward
        self.node_Vplus = R_in.nodeBackward
        self.node_Vminus = R_in.nodeForward
        
    

class Circuit:
    def __init__(self):
        self.components = []
        self.nodes = set()
    
    def addComponent(self, component):
        self.components.append(component)
        self.nodes.add(component.nodeForward)
        self.nodes.add(component.nodeBackward)
    def connectComponents(self, comp1, comp2):
        # Create merged node and get old nodes
        merged_node = Node()
        removed_node1 = comp1.nodeForward
        removed_node2 = comp2.nodeBackward
        
        # Merge connected components
        merged_node.connectedComponents = (
            removed_node1.connectedComponents +
            removed_node2.connectedComponents
        )
        
        # Update all components' node references
        for component in merged_node.connectedComponents:
            # Update forward references
            if component.nodeForward in {removed_node1, removed_node2}:
                component.nodeForward = merged_node
            # Update backward references
            if component.nodeBackward in {removed_node1, removed_node2}:
                component.nodeBackward = merged_node
        
        # Update circuit nodes
        self.nodes -= {removed_node1, removed_node2}
        self.nodes.add(merged_node)

    def setCircuitFrequency(self, omega):
        self.angularFrequency = omega
        for component in self.components:
            component.setFrequency(omega)

        
    def solveSystem(self):
        impedances = [comp.impedance for comp in self.components]
        
        # Get reference node (ground)
        largest_vsrc = self._get_largest_vsource()
        if largest_vsrc:
            ref_node = largest_vsrc.nodeBackward
        else:  # Fallback if no voltage sources
            ref_node = next(iter(self.nodes)) if self.nodes else None
            if not ref_node:
                raise ValueError("No nodes in circuit")
        
        non_ref_nodes = [n for n in self.nodes if n != ref_node]
        num_nodes = len(non_ref_nodes)
        
        # Count voltage sources
        v_sources = [c for c in self.components if isinstance(c, VoltageSource) and not isinstance(c, DependentVoltageSource)]
        num_v_sources = len(v_sources)
        
        dep_sources = [c for c in self.components if isinstance(c, DependentVoltageSource)]
        num_d_sources = len(dep_sources)
    
        # create matrix
        size = num_nodes + num_v_sources + num_d_sources
        A = np.zeros((size, size), dtype=complex)
        b = np.zeros(size, dtype=complex)
        
        # Create node index mapping
        node_indices = {node: idx for idx, node in enumerate(non_ref_nodes)}
        
        # Process regular nodes (KCL equations)
        for node_idx, node in enumerate(non_ref_nodes):
            for comp in node.connectedComponents:
                # Determine connected node
                other_node = comp.nodeForward if comp.nodeBackward == node else comp.nodeBackward
                
                # Handle component contribution
                if isinstance(comp, (Resistor, Capacitor)):
                    admittance = 1/comp.impedance if comp.impedance != 0 else 0
                    
                    A[node_idx, node_idx] += admittance
                    
                    # Mutual admittance if not reference
                    if other_node in node_indices:
                        A[node_idx, node_indices[other_node]] -= admittance
                        
                elif isinstance(comp, VoltageSource):
                    pass
        
        # Add voltage source equations
        for vsrc_idx, vsrc in enumerate(v_sources):
            row = num_nodes + vsrc_idx
            
            n_plus = vsrc.nodeForward
            n_minus = vsrc.nodeBackward
        
            if n_plus in node_indices:
                A[row, node_indices[n_plus]] = 1
            if n_minus in node_indices:
                A[row, node_indices[n_minus]] = -1
            b[row] = vsrc.value
 
            current_var_col = num_nodes + vsrc_idx

            if n_plus in node_indices:
                A[node_indices[n_plus], current_var_col] += 1
            if n_minus in node_indices:
                A[node_indices[n_minus], current_var_col] -= 1

        for dsrc_idx, dsrc in enumerate(dep_sources):
            row = num_nodes + num_v_sources + dsrc_idx
            n_plus = dsrc.nodeForward
            n_minus = dsrc.nodeBackward

            # Build the dependent source constraint: V(n_plus) - V(n_minus) - gain*(V(n1) - V(n2)) = 0
            if n_plus in node_indices:
                A[row, node_indices[n_plus]] += 1
            if n_minus in node_indices:
                A[row, node_indices[n_minus]] -= 1

            gain = dsrc.value
            n1 = dsrc.comp.nodeForward
            n2 = dsrc.comp.nodeBackward
            # Note: subtract the controlling voltage difference
            if n1 in node_indices:
                A[row, node_indices[n1]] += gain
            if n2 in node_indices:
                A[row, node_indices[n2]] += -gain

            # Add the stamp for the current variable of the dependent source (similar to independent voltage sources)
            current_var_col = num_nodes + num_v_sources + dsrc_idx
            if n_plus in node_indices:
                A[node_indices[n_plus], current_var_col] += 1
            if n_minus in node_indices:
                A[node_indices[n_minus], current_var_col] -= 1

     
        x = np.linalg.solve(A, b)
        
        # Update variables
        for node, idx in node_indices.items():
            node.voltage = x[idx]
        
        for vsrc_idx, vsrc in enumerate(v_sources):
            vsrc.current = x[num_nodes + vsrc_idx] 
            
        for dsrc_idx, dsrc in enumerate(dep_sources):
            dsrc.current = x[num_nodes + num_v_sources + dsrc_idx]       
        
        for comp in self.components:
            if isinstance(comp, (Resistor, Capacitor)):
                try:
                    V_plus = comp.nodeForward.voltage
                    V_minus = comp.nodeBackward.voltage
                    
                    # Handle floating components (both nodes non-reference)
                    if comp.nodeForward != ref_node and comp.nodeBackward != ref_node:
                        comp.current = (V_plus - V_minus) / comp.impedance
                    else:
                        comp.current = (V_plus - V_minus) / comp.impedance
                        
                except ZeroDivisionError:
                    comp.current = float('inf')  # Handle short circuits
                    print(f"Warning: Zero impedance in {comp.componentName}")
            
            
            
    def _get_largest_vsource(self):
        max_voltage = -float('inf')
        largest_vsrc = None
        for comp in self.components:
            if isinstance(comp, VoltageSource) and comp.value > max_voltage:
                max_voltage = comp.value
                largest_vsrc = comp
        return largest_vsrc

