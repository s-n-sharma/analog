* Band-Pass Filter
* Parameters: R1, C1, R2, C2
* Interface: in, out
* High-pass cutoff = 1/(2πR1C1)
* Low-pass cutoff = 1/(2πR2C2)

* High-pass section
C1 in 2 C1
R1 2 0 R1

* Low-pass section
R2 2 out R2
C2 out 0 C2 