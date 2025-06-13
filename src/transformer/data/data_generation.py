import random
from scipy.cluster.hierarchy import DisjointSet as djs
import translator
import math

class CircuitGeneration:
    func_num = 8
    common_resistor_values_ohms = [10, 22, 33, 47, 68, 100, 220, 330, 470, 680, 1000, 2200, 3300, 4700, 6800, 10000, 22000, 33000, 47000, 68000, 100000, 220000, 330000, 470000, 680000, 1000000]
    crv = common_resistor_values_ohms
    common_capacitor_values_farads = [10e-12, 22e-12, 33e-12, 47e-12, 68e-12, 100e-12, 220e-12, 330e-12, 470e-12, 680e-12, 1e-9, 2.2e-9, 3.3e-9, 4.7e-9, 6.8e-9, 10e-9, 22e-9, 33e-9, 47e-9, 68e-9, 100e-9, 220e-9, 330e-9, 470e-9, 680e-9, 1e-6, 2.2e-6, 3.3e-6, 4.7e-6, 10e-6, 22e-6, 33e-6, 47e-6, 100e-6, 220e-6, 330e-6, 470e-6, 1000e-6]
    ccv = common_capacitor_values_farads
    amt = 2
    def __init__(self, number_of_functions=3):
        self.number_of_functions = number_of_functions # max number of randomly generated function in a single netlist
    
    def min_deg_func(dic, zero_flag = False):
        for i in dic.keys():
            if zero_flag and i == 0:
                continue
            if dic[i] < 3:
                return False
        return True
    
    def getCircuit(self):
        node_number = random.randint(int(0.25*self.number_of_functions), int(1.25 * self.number_of_functions))
        djs_conn = djs([j for j in range(1, node_number+1)])
        min_deg_2 = False
        deg_count = {k : 0 for k in range(node_number+1)}
        Resistor_count = 0
        Capacitor_count = 0

        with open('temp.txt', 'w') as f:
            f.write(f"Vin 0 1 1\n")

            while not (len(djs_conn.subsets()) == 1) or not min_deg_2:
                
                comp_category = random.randint(0, 10)

                if comp_category < 4:
                    comp_type = 'R'
                elif comp_category < 8:
                    comp_type = 'C'
                else:
                    comp_type = 'F'
                
                node_1 = random.randint(1, node_number)
                node_2 = random.randint(1, node_number)

                while (node_1 == node_2):
                    node_2 = random.randint(1, node_number)
                
                djs_conn.merge(node_1, node_2)
                
                deg_count[node_1] += 1
                deg_count[node_2] += 1

                comp_strengths = [0.1, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 250, 375, 500, 1000, 5000, 10000]

                if comp_type == 'R':
                    Resistor_count += 1
                    f.write(f'R{Resistor_count} {node_1} {node_2} {random.choice(comp_strengths)}k\n')
                elif comp_type == 'C':
                    Capacitor_count += 1
                    f.write(f'C{Capacitor_count} {node_1} {node_2} {random.choice(comp_strengths)}n\n')
                else:
                    func_type = random.randint(1, self.func_num)

                    if func_type == 1:
                        vals = random.choices(comp_strengths, 4)
                        f.write(f'.fn {node_1} {node_2} bandpass R1={vals[0]}k C1={vals[1]}n R2={vals[2]}k C2={vals[3]}n\n')
                    elif func_type == 2:
                        vals = random.choices(comp_strengths, 2)
                        f.write(f'.fn {node_1} {node_2} inverting_amp R1={vals[0]}k R2={vals[1]}k\n')
                    elif func_type == 3:
                        vals = random.choices(comp_strengths, 2)
                        f.write(f'.fn {node_1} {node_2} non_inverting_amp R1={vals[0]}k R2={vals[1]}k\n')
                    elif func_type == 4:
                        vals = random.choices(comp_strengths, 2)
                        f.write(f'.fn {node_1} {node_2} rc_highpass R1={vals[0]}k C1={vals[1]}n')
                    elif func_type == 5:
                        vals = random.choices(comp_strengths, 2)
                        f.write(f'.fn {node_1} {node_2} rc_lowpass R1={vals[0]}k C1={vals[1]}n')
                    elif func_type == 6:
                        vals = random.choices(comp_strengths, 4)
                        f.write(f'.fn {node_1} {node_2} sallen_key_highpass R1={vals[0]}k R2={vals[1]}k C1={vals[2]}n C2={vals[3]}n\n')
                    elif func_type == 7:
                        vals = random.choices(comp_strengths, 4)
                        f.write(f'.fn {node_1} {node_2} sallen_key_lowpass R1={vals[0]}k R2={vals[1]}k C1={vals[2]}n C2={vals[3]}n\n')
                    else:
                        vals = random.choices(comp_strengths, 2)
                        f.write(f'.fn {node_1} {node_2} voltage_divider R1={vals[0]} R2={vals[1]}')


                min_deg_2 = self.min_deg_func(deg_count, True)
            
            #connections to ground
            ground_conns_count = random.randint(1, int(0.2*node_number))

            for _ in range(ground_conns_count):
                comp_category = random.randint(0, 10)

                if comp_category < 4:
                    comp_type = 'R'
                elif comp_category < 8:
                    comp_type = 'C'
                else:
                    comp_type = 'F'
                
                node_1 = 0
                node_2 = random.randint(1, node_number)

                comp_strengths = [0.1, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 250, 375, 500, 1000, 5000, 10000]

                if comp_type == 'R':
                    Resistor_count += 1
                    f.write(f'R{Resistor_count} {node_1} {node_2} {random.choice(comp_strengths)}k\n')
                elif comp_type == 'C':
                    Capacitor_count += 1
                    f.write(f'C{Capacitor_count} {node_1} {node_2} {random.choice(comp_strengths)}n\n')
                else:
                    func_type = random.randint(1, self.func_num)

                    if func_type == 1:
                        vals = random.choices(comp_strengths, 4)
                        f.write(f'.fn {node_1} {node_2} bandpass R1={vals[0]}k C1={vals[1]}n R2={vals[2]}k C2={vals[3]}n\n')
                    elif func_type == 2:
                        vals = random.choices(comp_strengths, 2)
                        f.write(f'.fn {node_1} {node_2} inverting_amp R1={vals[0]}k R2={vals[1]}k\n')
                    elif func_type == 3:
                        vals = random.choices(comp_strengths, 2)
                        f.write(f'.fn {node_1} {node_2} non_inverting_amp R1={vals[0]}k R2={vals[1]}k\n')
                    elif func_type == 4:
                        vals = random.choices(comp_strengths, 2)
                        f.write(f'.fn {node_1} {node_2} rc_highpass R1={vals[0]}k C1={vals[1]}n')
                    elif func_type == 5:
                        vals = random.choices(comp_strengths, 2)
                        f.write(f'.fn {node_1} {node_2} rc_lowpass R1={vals[0]}k C1={vals[1]}n')
                    elif func_type == 6:
                        vals = random.choices(comp_strengths, 4)
                        f.write(f'.fn {node_1} {node_2} sallen_key_highpass R1={vals[0]}k R2={vals[1]}k C1={vals[2]}n C2={vals[3]}n\n')
                    elif func_type == 7:
                        vals = random.choices(comp_strengths, 4)
                        f.write(f'.fn {node_1} {node_2} sallen_key_lowpass R1={vals[0]}k R2={vals[1]}k C1={vals[2]}n C2={vals[3]}n\n')
                    else:
                        vals = random.choices(comp_strengths, 2)
                        f.write(f'.fn {node_1} {node_2} voltage_divider R1={vals[0]} R2={vals[1]}')
            
            ground_node = random.randint(1, node_number)

            f.write(f'Vout {ground_node}\n')
            f.write('.end')

        trans = translator.Translator('temp.txt')
        return trans.circuit
    
    compIDs = {0 : 'R', 1 : 'C', 2 : 'bandpass', 3 : 'inverting_amp', 4 : 'non_inverting amp', 5 : 'rc_highpass', 6 : 'rc_lowpass', 7 : 'sallen_key_highpass', 8 : 'sallen_key_lowpass', 9: 'voltage_divider'}
    

    def checker(conn, deg):

        for subs in conn.subsets():
            if not len(subs):
                continue
            count = 0
            for node in subs:
                if deg[node] < 2:
                    count += 1
        
            if count > 2:
                return False
        
        return True

            

    def getCircuit2(self):

        intermediate_node_number = 0

        while (True):

            for _ in range(math.factorial(max(10, intermediate_node_number - 2))):

                conn = djs([k for k in range(2, intermediate_node_number+2)]) #intermediate nodes only
                deg_check = {k : 0 for k in range(2, intermediate_node_number + 2)}

                #first stage is to determine the layout of the circuit (what connects each node), this is randomly generated

                #contains data of the form of a list of [compID, node_1, node_2]
                #later append list of component values
                components = [] 

                inter_comp_edge_count = random.randint(0, int(1.5 * intermediate_node_number))

                i = 0

                while i < inter_comp_edge_count or not self.checker(conn, deg_check):
                    
                    node_1 = node_2 = random.randint(2, intermediate_node_number + 1)
                    
                    while (node_2 == node_1):
                        node_2 = random.randint(2, intermediate_node_number + 1)
                    

                    comp_seed = random.randint(-7, 9)

                    if comp_seed <= 1:
                        comp_ID = comp_seed % 2
                    else:
                        comp_ID = comp_seed
                    
                    deg_check[node_1] += 1
                    deg_check[node_2] += 2

                    conn.merge(node_1, node_2)

                    components.append([comp_ID, node_1, node_2])

                    i += 1
                
                #now we connect each of the subsets to the Vin and Ground

                for subs in conn.subsets():

                    nodes = []

                    for node in subs:

                        if deg_check[node] == 1:

                            nodes.append(node)
                    
                    node_1, node_2 = nodes[0], nodes[1]

                    if not random.randint(0, 1):
                        #node_1 connected to Vin

                        number_of_conn = random.randint(1, 3)

                        for nc in range(number_of_conn):

                            comp_seed = random.randint(-7, 9)

                            if comp_seed <= 1:
                                comp_ID = comp_seed % 2
                            else:
                                comp_ID = comp_seed
                            
                            components.append([comp_ID, 1, node_1])

                        number_of_conn = random.randint(1, 3)

                        for nc in range(number_of_conn):

                            number_of_conn = random.randint(1, 3)

                        for nc in range(number_of_conn):

                            comp_seed = random.randint(-7, 9)

                            if comp_seed <= 1:
                                comp_ID = comp_seed % 2
                            else:
                                comp_ID = comp_seed
                            
                            components.append([comp_ID, 0, node_2])
                    else:

                        number_of_conn = random.randint(1, 3)

                        for nc in range(number_of_conn):

                            comp_seed = random.randint(-7, 9)

                            if comp_seed <= 1:
                                comp_ID = comp_seed % 2
                            else:
                                comp_ID = comp_seed
                            
                            components.append([comp_ID, 1, node_2])

                        number_of_conn = random.randint(1, 3)

                        for nc in range(number_of_conn):

                            number_of_conn = random.randint(1, 3)

                        for nc in range(number_of_conn):

                            comp_seed = random.randint(-7, 9)

                            if comp_seed <= 1:
                                comp_ID = comp_seed % 2
                            else:
                                comp_ID = comp_seed
                            
                            components.append([comp_ID, 0, node_1])

                #the second is to assign values to each of the components in the circuit

                part = []

                r_count = 0
                c_count = 0
                tc = 0

                for comp in components:

                    if comp[0] == 0:
                        r_count += 1
                        comp.append([f'R{r_count}'])
                        comp.append([random.sample(self.crv, self.amt)])
                        tc += 1

                    if comp[0] == 1:
                        c_count += 1
                        comp.append([f'C{c_count}'])
                        comp.append([random.sample(self.crv, self.amt)])
                        tc += 1

                    
                    if comp[0] == 2: #bandpass
                        comp.append(['R1, C1, R2, C2'])
                        comp.append([random.sample(self.crv, self.amt), random.sample(self.ccv, self.amt), random.sample(self.crv, self.amt), random.sample(self.ccv, self.amt)])
                        tc += 1


                    if comp[0] == 3: #inverting_amp
                        comp.append(['R1', 'R2'])
                        comp.append([random.sample(self.crv, self.amt), random.sample(self.crv, self.amt)])
                        tc += 1

                    
                    if comp[0] == 4: #non_inverting_amp
                        comp.append(['R1', 'R2'])
                        comp.append([random.sample(self.crv, self.amt), random.sample(self.crv, self.amt)])
                        tc += 1

                    
                    if comp[0] == 5: #rc_highpass
                        comp.append(['R', 'C'])
                        comp.append([random.sample(self.crv, self.amt), random.sample(self.crv, self.amt)])
                        tc += 1

                    
                    if comp[0] == 6: #rc_lowpass
                        comp.append(['R', 'C'])
                        comp.append([random.sample(self.crv, self.amt), random.sample(self.crv, self.amt)])
                        tc += 1

                    
                    if comp[0] == 7: #sallen_key_highpass
                        comp.append(['R1', 'R2', 'C1', 'C2'])
                        comp.append([random.sample(self.crv, self.amt), random.sample(self.crv, self.amt), random.sample(self.ccv, self.amt), random.sample(self.ccv, self.amt)])
                        tc += 1

                    
                    if comp[0] == 8: 
                        comp.append(['R1', 'R2', 'C1', 'C2'])
                        comp.append([random.sample(self.crv, self.amt), random.sample(self.crv, self.amt), random.sample(self.ccv, self.amt), random.sample(self.ccv, self.amt)])
                        tc += 1

                    
                    if comp[0] == 9:
                        comp.append(['R1', 'R2'])
                        comp.append([random.sample(self.crv, self.amt), random.sample(self.crv, self.amt)])
                        tc += 1


                        


                



                        










