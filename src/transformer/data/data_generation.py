import random
from scipy.cluster.hierarchy import DisjointSet as djs
import translator

class CircuitGeneration:
    func_num = 8
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
        






