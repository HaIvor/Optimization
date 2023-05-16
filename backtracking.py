import numpy as np 
import numdifftools as nd 
import matplotlib.pyplot as plt
import sympy as sym
import time
start_time = time.time()


xi = np.array([
    [10],
    [9.5]
])

def calculateHessianAndGradient(xi):

    #Defining 2 variables
    x1, x2 = sym.symbols('x1 x2')

    # Defining the function we want to minimize
    function = (1-x1)**2 + 100 * (x2 - x1**2)**2

    #function value
    function_value = function.evalf(subs={x1: xi[0][0], x2: xi[1][0]}) 

    # Derivating f(x) for x1, x2, x3 (algebraic answer, without values)
    der_x1 = function.diff(x1)
    der_x2 = function.diff(x2)

    # Putting values into the derivatives
    der_x1_values = function.diff(x1).evalf(subs={x1: xi[0][0], x2: xi[1][0]}) 
    der_x2_values = function.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0]}) 

    # Putting the derivatives together in a matrix so we get the gradient of the objective function
    gradient_values = np.array([
        [der_x1_values],
        [der_x2_values]
    ], dtype=np.float32)

    # Derivating the objective function further to get the hessian
    der_x1x1_values = der_x1.diff(x1).evalf(subs={x1: xi[0][0], x2: xi[1][0]}) 
    der_crossx1x2_values = der_x1.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0]}) 
    der_x2x2_values = der_x2.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0]}) 

    # Putting the derivatives together in a matrix so we get the hessian of the objective function
    hessian_values = np.array([
        [der_x1x1_values, der_crossx1x2_values],
        [der_crossx1x2_values, der_x2x2_values],
    ], dtype=np.float32)

    # Backtracking start 
    epsilon = 1
    
    # Newton's method direction
    deltax = -1.0*np.linalg.inv(hessian_values)@gradient_values

    xEpsilondeltax = xi + epsilon*deltax
    f_xEpsilondeltax = function.evalf(subs={x1: xEpsilondeltax[0][0], x2: xEpsilondeltax[1][0]})

    while f_xEpsilondeltax > function_value+alpha*epsilon*np.transpose(gradient_values)@deltax:
        xEpsilondeltax = xi + epsilon*deltax
        f_xEpsilondeltax = function.evalf(subs={x1: xEpsilondeltax[0][0], x2: xEpsilondeltax[1][0]})
        epsilon = beta*epsilon
    # backtracking finish

    return gradient_values, hessian_values, epsilon

tolerance = 1e-5 # How exact one wants the end result to be (lower epsilon => more exact)
i = 1 # main loop iteration start
x1_list = np.array(xi[0]) # Used for plotting
x2_list = np.array(xi[1]) # Used for plotting

NR_iteration = 1 # All the Newton's method interations, used for plotting x-axis

# Backtracking values
alpha = 0.3
beta = 0.7

# Initialize the Newton decrement squared
lambda_2 = 1

# The stopping criterion has something to do with the Newton decrement. More information: "Convex Optimization", Stephen Boyd & Lieven Vandenberghe. p. 486
while (lambda_2/2) > tolerance:
    
    # Pulling out the gradient/hessian at point xi + epsilon that has gone through the backtracking algorithm
    gradient, hessian, epsilon = calculateHessianAndGradient(xi)

    # The Newton decrement squared = lamba**2 = lambda_2
    lambda_2 = np.transpose(gradient)@np.linalg.inv(hessian)@gradient

    # Uncomment to turn off backtracking
    #epsilon = 1

    # Checking how much the backtracking changes epsilon
    if epsilon != 1: 
        print("Backtracking changed epsilon to ---->: ", epsilon)

    # Newton's method
    xi = xi - epsilon*np.linalg.inv(hessian)@gradient

    print("xi: \n", xi)
    print("---")

    # For plotting
    x1_list = np.append(x1_list, xi[0])
    x2_list = np.append(x2_list, xi[1])
    NR_iteration += 1

print("=====DONE NR=====")

# Plotting stuff 
figure, (ax1, ax2) = plt.subplots(1,2, figsize=(14, 8))

ax1.plot(np.arange(NR_iteration), x1_list, "g")
ax2.plot(np.arange(NR_iteration), x2_list, "r")

ax1.grid()
ax2.grid()

ax1.set_xlabel('iterations')
ax2.set_xlabel('iterations')

ax1.set_title('$x_1$')
ax2.set_title('$x_2$')

figure.suptitle('Backtracking on the "Rosenbrock Function"', fontsize=16)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()