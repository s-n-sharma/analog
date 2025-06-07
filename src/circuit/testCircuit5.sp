* Test circuit demonstrating functional circuits
* Using active filter with different parameter sets
V1 1 0 1

* First instance of active filter
.fn 1 2 active_filter R1=10k R2=20k C1=1u C2=2u

* Second instance with different values
* First instance of active filter
.fn 2 3 active_filter R1=20k R2=40k C1=3u C2=6u

X 3 4 NESTED
* Input voltage source

* Output node
VOUT 4