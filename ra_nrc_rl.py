# Pseudo code for node i 

import random 
import numpy as np 
import numdifftools as nd
import matplotlib.pyplot as plt
import sympy as sym
import sys 

"""
The values we are actually optimizing: 

N = optimizable - Subcarriers 
M = optimizable - Modulation order
m = optimizable - Symbols per packet
"""

def calculateHessianAndGradient(xi):

    R_c = 0.5 # Coding rate
    B = 12000 # Bandwidth
    p_c = 0.25 # Cyclic prefix fraction
    t_oh = 0.1 # Overhead (preamble, propagation...)
    R_n = 0.5 # 0<R_n<1 A part of the carrier waves 
    r = 1000 # r = 100-4000m -emil
    c1 = 1500 # Speed of sound, 343 m/s
    t_d = r/c1 # Transmission delay t_d = r/c
    p_lt = 0.001 # Target packet loss ratio
    gamma = 31.62 # The average SNR at the receiver, 0 < gamma_dB < 30

    N, M, m = sym.symbols('N M m')

    # Constraints for the cost function. constraint_N/M/m is really 2 constraints. This means we have 7 constraints here.
    constraint_N = sym.log(-(N - 2000)) + sym.log(-(400 - N)) # 400 < N < 2000
    constraint_M = sym.log(-(2 - M)) + sym.log(-(M - 64)) # 2 < M < 64
    constraint_m = sym.log(-(1 - m)) + sym.log(-(m - 40)) #  1 < m < 40
    constraint_PLR = sym.log(-(sym.log(m)+sym.log(R_n) + sym.log(N) + sym.log(sym.log(M,2)) + (1/R_c)*(sym.log(0.2)-(3*gamma)/(2*(M-1))) - sym.log(p_lt) )  ) # PLR - packet loss ratio

    # Defining the cost function
    function_without_constraint = -(sym.log(m)+ sym.log(R_c) + sym.log(B) + sym.log(R_n) + sym.log(N) + sym.log(sym.log(M, 2)) - sym.log(m*(1+p_c)*N + B*(t_oh + t_d)))
    
    # Defining the objective function we want to minimize, notice using the bb notation instead of t
    function = bb*( function_without_constraint ) - (constraint_N+constraint_M+constraint_m+constraint_PLR) 

    # Objective function value at xi
    function_value = function.evalf(subs={N: xi[0][0], M: xi[1][0], m: xi[2][0]})

    # Derivating the objective function for N, M and m (algebraic answer, without values)
    der_x1 = function.diff(N)
    der_x2 = function.diff(M)
    der_x3 = function.diff(m)

    # Putting xi values into the derivatives
    der_x1_values = function.diff(N).evalf(subs={N: xi[0][0], M: xi[1][0], m: xi[2][0]})
    der_x2_values = function.diff(M).evalf(subs={N: xi[0][0], M: xi[1][0], m: xi[2][0]})
    der_x3_values = function.diff(m).evalf(subs={N: xi[0][0], M: xi[1][0], m: xi[2][0]})

    # Putting the derivatives together in a matrix so we get the gradient of the objective function
    gradient_values = np.array([
        [der_x1_values],
        [der_x2_values],
        [der_x3_values]
    ], dtype=np.float32)

    # Derivating the objective function further to get the hessian
    der_x1x1_values = der_x1.diff(N).evalf(subs={N: xi[0][0], M: xi[1][0], m: xi[2][0]})
    der_crossx1x2_values = der_x1.diff(M).evalf(subs={N: xi[0][0], M: xi[1][0], m: xi[2][0]})
    der_x2x2_values = der_x2.diff(M).evalf(subs={N: xi[0][0], M: xi[1][0], m: xi[2][0]})
    der_crossx1x3_values = der_x1.diff(m).evalf(subs={N: xi[0][0], M: xi[1][0], m: xi[2][0]})
    der_x3x3_values = der_x3.diff(m).evalf(subs={N: xi[0][0], M: xi[1][0], m: xi[2][0]})
    der_crossx2x3_values = der_x2.diff(m).evalf(subs={N: xi[0][0], M: xi[1][0], m: xi[2][0]})

    # Putting the derivatives together in a matrix so we get the hessian of the objective function
    hessian_values = np.array([
        [der_x1x1_values, der_crossx1x2_values, der_crossx1x3_values],
        [der_crossx1x2_values, der_x2x2_values, der_crossx2x3_values],
        [der_crossx1x3_values, der_crossx2x3_values, der_x3x3_values]
    ], dtype=np.float32)

    # Check if the answer is feasible. For example a too big starting t value can cause numerical difficulties 
    if sym.im(function_value) != 0:
        print('=========COMPLEX ANSWER, NOT FEASIBLE, MAYBE DECREASE t =============')
        print("f(x): ", function_value)
        sys.exit()

    return gradient_values, hessian_values

# dimension of hessian / gradient. ex. 3x3 and 3x1
n = 3 

# Number of out neighbors. Since this simulation is just using two nodes, they each have one neighbor.
out_neigh = 1 

# Value used in order to increase the basin of attraction and the robustness of the algorithm
c = 1E-8

# How exact one wants before the simulation stops. A lower tolerance value => more exact
tolerance = 1e-3

# We set a very low epsilon value first iteration to not get a big jump in the beginning
epsilon = 0.00001

# The bb value is also used as the "t" parameter. This is the parameter used in the barrier method / interior point method.
# It is important to not start with a too big bb value. 
bb = 1

# Max iterations (if it never converges)
max_iter = 500

# Note: The next steps heavily follows the given pseudocode in the bachelor report.

# -----------------Initialization node i--------------------

# Initial estimate for node i for the global optimization. Arbitrary values.
xi = np.array([
    [500], # N_start
    [4], # M_start
    [5] # m_start
])

yi, gi, gi_old = np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)) # if n=3 -> they are 3x1 zero matrices
zi, hi, hi_old = np.identity(n), np.identity(n), np.identity(n) # if n=3 -> they are 3x3 identity matrices
sigmai_y, sigmai_z = np.zeros((n, 1)), np.zeros((n, n)) # 3x1 and 3x3 filled with zeros. Gradient is 3x1 and hessian is 3x3
sigmaj_y, sigmaj_z = np.zeros((n, 1)), np.zeros((n, n)) 
rhoi_y, rhoi_z = np.zeros((n,1)), np.zeros((n,n))

# ------------------Initialization node j------------------------

# Initial estimate for node j for the global optimization. Arbitrary values.
xj = np.array([ 
    [500], # N_start
    [4], # M_start
    [5] # m_start
]) 

yj, gj, gj_old = np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)) # if n=3 -> they are 3x1 zero matrices
zj, hj, hj_old = np.identity(n), np.identity(n), np.identity(n) # if n=3 -> they are 3x3 identity matrices
sigmaj_y, sigmaj_z = np.zeros((n, 1)), np.zeros((n, n)) # 3x1 and 3x3. Gradient is 3x1 and hessian is 3x3
rhoj_y, rhoj_z = np.zeros((n,1)), np.zeros((n,n))

flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
flag_update = 1

# Used for when plotting N, M and m.
#node i
N_list = np.array(xi[0])
M_list = np.array(xi[1])
m_list = np.array(xi[2])
#node j
N_list2 = np.array(xj[0])
M_list2 = np.array(xj[1])
m_list2 = np.array(xj[2])
i = 1
#---------------------------------------------------------------------------------------

while i < max_iter: 
# ------Data transmission node i-------

    if flag_transmission == 1:
        transmitter_node_ID = "i" # This is what will be transmitted (but not really in this simulation)
        
        # Push sum consensus
        yi = (1/(out_neigh + 1))*yi
        zi = (1/(out_neigh + 1))*zi
        
        # These sigmas are broadcasted to the neighbors (just setting them global in the simulation)
        sigmai_y = sigmai_y + yi
        sigmai_z = sigmai_z + zi

        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_reception2 = 1
        
# ------Data Reception node i------

    if flag_reception == 1:

        # Should get these values from node j: transmitter_node_ID, sigmaj_y and sigmaj_z 

        yi = yi + np.float16(sigmaj_y) - rhoi_y
        zi = zi + np.float16(sigmaj_z) - rhoi_z

        rhoi_y = np.float16(sigmaj_y)
        rhoi_z = np.float16(sigmaj_z) 

        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_update = 1
        

# ------Estimate Update node i------ 

    if flag_update == 1:
        

        # In order to increase the basin of attraction and the robustness of the algorithm, it is suitable to force: hessian >= c*I
        if (np.abs(np.linalg.eigvals(zi)) < c).all():
            zi = c*np.identity(n)

        # Newton-Raphson Consensus
        xi = (1-epsilon)*xi + epsilon*np.linalg.inv(zi)@yi 
        
        # For plotting the values
        N_list = np.append(N_list, xi[0][0])
        M_list = np.append(M_list, xi[1][0])
        m_list = np.append(m_list, xi[2][0])
        
        gi_old = gi
        hi_old = hi
        gradient, hi = calculateHessianAndGradient(xi)
        gi = hi@xi-gradient

        # For debugging
        print("yi ", yi)
        print("zi ", zi)
        print("xi: \n", xi)

        yi = yi + gi - gi_old
        zi = zi + hi - hi_old

        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_transmission = 1
        print("iteration: ", i)

        # For plotting and simulating
        i += 1 
        
        # Stopping criterion, it stops when xi and xj are approximately the same.
        # Can decrease tolerance to make it even more accurate
        diff = np.abs(xi - xj)
        if np.all(diff < tolerance):
            break
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

    # ------Data transmission node j-------

    if flag_transmission2 == 1:
        transmitter_node_ID = 2 # transmitter_node_ID = i, just saying we are working on node 1 now ex.
        
        # Push sum consensus
        yj = (1/(out_neigh + 1))*yj
        zj = (1/(out_neigh + 1))*zj

        # The sigmas are broadcasted to the neighbors + the node_ID
        sigmaj_y = sigmaj_y + yj
        sigmaj_z = sigmaj_z + zj

        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_reception = 1

    # ------Data Reception node j------

    if flag_reception2 == 1:

        # Should get values from node j: transmitter_node_ID, sigmaj_y and sigmaj_z

        yj = yj + np.float16(sigmai_y) - rhoj_y 
        zj = zj + np.float16(sigmai_z) - rhoj_z

        rhoj_y = np.float16(sigmai_y)
        rhoj_z = np.float16(sigmai_z) 

        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_update2 = 1

    # ------Estimate Update 2------ 

    if flag_update2 == 1:

        # In order to increase the basin of attraction and the robustness of the algorithm, it is suitable to force: hessian >= c*I
        if (np.abs(np.linalg.eigvals(zj)) < c).all():
            zj = c*np.identity(n)
        
        # Newton's method
        xj = (1-epsilon)*xj + epsilon*np.linalg.inv(zj)@yj # Newton-Raphson Consensus

        # This will be the actual step size further in the simulation
        epsilon = 0.2

        # For plotting the values
        N_list2 = np.append(N_list2, xj[0][0])
        M_list2 = np.append(M_list2, xj[1][0])
        m_list2 = np.append(m_list2, xj[2][0])
        
        gj_old = gj
        hj_old = hj
        gradient, hj = calculateHessianAndGradient(xj)
        gj = hj@xj-gradient

        yj = yj + gj - gj_old
        zj = zj + hj - hj_old

        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_transmission2 = 1

# ------------------------Plotting------------------------------

figure, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10, 8))

ax1.plot(np.arange(i), N_list, "b", label="N - node 1")
ax1.plot(np.arange(i-1), N_list2, "c", label="N - node 2")

ax2.plot(np.arange(i), M_list, "r", label="M - node 1")
ax2.plot(np.arange(i-1), M_list2, "y", label="M - node 2")

ax3.plot(np.arange(i), m_list, "g", label="m - node 1")
ax3.plot(np.arange(i-1), m_list2, "k", label="m - node 2")

ax1.grid()
ax2.grid()
ax3.grid()

ax1.legend(loc="lower right")
ax2.legend(loc="lower right")
ax3.legend(loc="lower right")

ax1.set_xlabel('Iterations')
ax2.set_xlabel('Iterations')
ax3.set_xlabel('Iterations')

figure.suptitle('ra-NRC with two nodes (float16 restriction + constant IPM parameter value) - Forced sequence', fontsize=16)

print("---------------------------------------------------------------------")
print(f"The last N values: \n N1 = {N_list[-1]} \n N2 = {N_list2[-1]}\n")
print(f"The last M values: \n M1 = {M_list[-1]} \n M2 = {M_list2[-1]} \n")
print(f"The last m values: \n m1 = {m_list[-1]} \n m2 = {m_list2[-1]}")
print("---------------------------------------------------------------------")

plt.show()
