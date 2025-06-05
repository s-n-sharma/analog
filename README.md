# analog

### How to write circuits

1. **Declare subcircuits at the top**  
   ```spice
   .subckt SK1 <node> <start_node_no.> <end_node_no.>
   ```

2. **Declare voltage sources** (any line that starts with `V`)  
   ```spice
   Vin <n_node_no.> <p_node_no.> <value>
   ```

3. **Declare components** (`R` = resistor, `C` = capacitor, `IOP` = Ideal op amp)  
   ```spice
   R1 <n_node_no.> <p_node_no.> <value>
   C1 <n_node_no.> <p_node_no.> <value>
   IOP <V+_no.> <V-_no.> <Vout_no.>
   ```

4. **Specify the output node**  
   ```spice
   Vout <n_node_no.> 0
   ```

5. **End the netlist**  
   ```spice
   .end
   ```

## How to write subcircuits

1. Same as above but before end line write:
   ```spice
   .declare_subckt <start_node_no.> <end_node_no.>
   ```

