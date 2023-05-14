import numpy as np
from scipy.optimize import minimize

log = np.log # The natural logarithm
log2 = np.log2 # Log base 2

# Values for the cost function

R_c = 0.5 # Coding rate
B = 12000 # Bandwidth
p_c = 0.25 # Cyclic prefix fraction
t_oh = 0.1 # Overhead (preamble, propagation...)
R_n = 0.5 # 0<R_n<1 A part of the carrier waves
r = 1000 # r = 100-4000m, range
c1 = 1500 # Speed of sound, 1500 m/s
t_d = r/c1 # Transmission delay t_d = r/c
p_lt = 0.001 # Target packet loss ratio
gamma = 31.62 # The average SNR at the receiver, 0<gamma_dB<30

# Starting values 
start_N = 600
start_M = 7
start_m = 5

# Defining the cost function
def f(x):
    return -(log(x[2])+log(R_c)+log(B)+log(R_n)+log(x[0])+log(log2(x[1]))-log(x[2]*(1+p_c)*x[0]+B*(t_oh+t_d)))

# Constraints. Defined as g_i(x) > 0 (opposite of barrier method)
cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 400},
        {'type': 'ineq', 'fun': lambda x: 2000 - x[0]},
        {'type': 'ineq', 'fun': lambda x: x[1] - 2},
        {'type': 'ineq', 'fun': lambda x: 64 - x[1]},
        {'type': 'ineq', 'fun': lambda x: x[2] - 1},
        {'type': 'ineq', 'fun': lambda x: 40 - x[2]},
        {'type': 'ineq', 'fun': lambda x: -(log(x[2])+log(R_n) + log(x[0]) + log(log2(x[1])) + (1/R_c)*(log(0.2)-(3*gamma)/(2*(x[1]-1))) - log(p_lt))})

# Minimize cost function with initial guess x0 and constraints
res = minimize(f, (start_N,start_M, start_m), constraints=cons, method="SLSQP") 
res2 = minimize(f, (start_N,start_M, start_m), constraints=cons, method="COBYLA")

print("--------------")
print("Optimal value, method: SLSQP")
print(f" N = {res.x[0]}, M = {res.x[1]}, m = {res.x[2]} \n\n with starting values = [{start_N}, {start_M}, {start_m}] and added constraints")
print(f"\n cost function is then equal to: {res.fun}")
print("--------------")

print("Optimal value, method: COBYLA")
print(f" N = {res2.x[0]}, M = {res2.x[1]}, m = {res2.x[2]} \n\n with starting values = [{start_N}, {start_M}, {start_m}] and added constraints")
print("\n cost function is then equal to: ", res2.fun)
print("===============DONE=================")
