# analog

### How to write circuits

1. **Declare voltage sources** (any line that starts with `V`)  
   ```spice
   Vin <n_node_no.> <p_node_no.> <value>
   ```

2. **Declare components** (`R` = resistor, `C` = capacitor, `IOP` = Ideal op amp, `X` = name of sub circuit)  
   ```spice
   R1 <n_node_no.> <p_node_no.> <value>
   C1 <n_node_no.> <p_node_no.> <value>
   IOP <V+_no.> <V-_no.> <Vout_no.>
   X <start_node> <end_node> name
   ```

3. **Specify the output node**  
   ```spice
   Vout <n_node_no.> 0
   ```
4. **Declare subcircuit if necessary**
   ```spice
   .declare_subckt <start_node> <end_node>
   ```
4. **End the netlist**  
   ```spice
   .end
   ```



