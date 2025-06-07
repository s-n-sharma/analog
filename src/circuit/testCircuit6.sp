Vin 0 1 1
.fn 1 2 voltage_divider R1=10k R2=20k
.fn 2 3 rc_lowpass R=1k C=1u
.fn 3 4 bandpass R1=1k C1=1u R2=10k C2=0.1u
.fn 4 5 non_inverting_amp R1=1k R2=10k
Vout 5
.end