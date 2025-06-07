# analog
## Circuit Blocks
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
   Vout <n_node_no.> 
   ```

4. **End the netlist**  
   ```spice
   .end
   ```
### Subcircuits
Fixed-component circuit blocks defined in `.subckts`:
```spice
* In .subckts/example.sp:
.declare_subckt in out
R1 in out 1k
R2 out 0 2k

* In main circuit:
X1 1 2 example
```

### Functions
Parameterized circuit blocks defined in `.functions`:
```spice
* In .functions/example.sp:
* Parameters: R1, R2
R1 in out R1
R2 out 0 R2

* In main circuit:
.fn 1 2 example R1=1k R2=2k
```

## Usage

1. **Subcircuits**: Use `X` prefix for fixed blocks
2. **Functions**: Use `.fn` for parameterized blocks
3. **Interface**: 
   - Subcircuits: `.declare_subckt in out`
   - Functions: Use `in` and `out` nodes

## Examples

### Cascaded Filters
```spice
* Input source
V1 1 0 1

* Low-pass filter
.fn 1 2 rc_lowpass R=1k C=1u

* Band-pass filter
.fn 2 3 bandpass R1=1k C1=1u R2=10k C2=0.1u

* Output amplifier
.fn 3 4 non_inverting_amp R1=1k R2=10k
```

### Mixed Usage
```spice
* Function with parameters
Vin 0 1 5

.fn 1 2 voltage_divider R1=10k R2=20k

* Fixed subcircuit
X1 2 3 fixed_filter

Vout 3
.end
```

## Best Practices

1. Document parameters and circuit behavior
2. Use standard units (k, u, n, p)
3. Test with various parameter values
4. Verify node connections

