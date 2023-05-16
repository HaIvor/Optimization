import random 
import numpy as np 
import numdifftools as nd
import matplotlib.pyplot as plt
import sympy as sym

def calculateHessianAndGradient(xi):

    #Defining 3 variables
    x1, x2, x3 = sym.symbols('x1 x2 x3')

    # Defining the objective function we want to minimize
    function = x1**2-500*x1+x2**2-30*x2+5*x3**2-60*x3

    #Defining the function value at xi
    function_value = function.evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]}) 

    # Derivating f(x) for x1, x2, x3 (algebraic answer, without values)
    der_x1 = function.diff(x1)
    der_x2 = function.diff(x2)
    der_x3 = function.diff(x3)

    # Putting values into the derivatives
    der_x1_values = function.diff(x1).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x2_values = function.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x3_values = function.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})

    # Putting the derivatives together in a matrix so we get the gradient of the objective function (3x1 matrix)
    gradient_values = np.array([
        [der_x1_values],
        [der_x2_values],
        [der_x3_values]
    ], dtype=np.float32)

    # Derivating the objective function further to get the hessian (3x3 matrix)
    der_x1x1_values = der_x1.diff(x1).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_crossx1x2_values = der_x1.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x2x2_values = der_x2.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_crossx1x3_values = der_x1.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x3x3_values = der_x3.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_crossx2x3_values = der_x2.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})

    # Putting the derivatives together in a matrix so we get the hessian of the objective function
    hessian_values = np.array([
        [der_x1x1_values, der_crossx1x2_values, der_crossx1x3_values],
        [der_crossx1x2_values, der_x2x2_values, der_crossx2x3_values],
        [der_crossx1x3_values, der_crossx2x3_values, der_x3x3_values]
    ], dtype=np.float32)
    
    return gradient_values, hessian_values, function_value

# Starting values
xi = np.array([ 
    [10], # x1 start
    [20], # x2 start 
    [70] # x3 start
]) 

# For plotting 
x1_list = np.array(xi[0])
x2_list = np.array(xi[1])
x3_list = np.array(xi[2])
cost_list = np.array([])
iterations = 1

# If the hessian misbehaves 
c = 1E-6 

# A value of how accurate one wants the end result to be. A smaller value = bigger precision 
accuracy = 1e-5

# Newton's method step size 
epsilon = 0.2

# For stopping criterion 
lambda_2 = 1

# The stopping criterion has something to do with the Newton decrement. More information: "Convex Optimization", Stephen Boyd & Lieven Vandenberghe. p. 486
while (lambda_2/2) > accuracy:

    # Pulling out the gradient/hessian at point xi
    gradient, hessian, function_value = calculateHessianAndGradient(xi)

    # The Newton decrement squared = lambda**2 = lambda_2
    lambda_2 = np.transpose(gradient)@np.linalg.inv(hessian)@gradient

    # In order to increase the basin of attraction and the robustness of the algorithm, it is suitable to force: hessian >= c*I
    if (np.abs(np.linalg.eigvals(hessian)) < c).all():
            hessian = c*np.identity(3) 

    print("hessian: \n", hessian)
    print("gradient: \n", gradient)

    # Newton's method
    xi = (1-epsilon)*xi+epsilon*(np.linalg.inv(hessian))@(hessian@xi-gradient)

    # For plotting
    x1_list = np.append(x1_list, xi[0])
    x2_list = np.append(x2_list, xi[1])
    x3_list = np.append(x3_list, xi[2])
    cost_list = np.append(cost_list, function_value)

    iterations = iterations + 1

print("=====DONE NR=====")

print("\n xi is then in the end: \n", xi)
print("\n cost function is: \n", function_value)

figure, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(14, 8))

ax1.plot(np.arange(iterations), x1_list, "g")
ax2.plot(np.arange(iterations), x2_list, "r")
ax3.plot(np.arange(iterations), x3_list, "b")
ax4.plot(np.arange(iterations-1), cost_list, "k")

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

ax1.set_xlabel('Iteration')
ax2.set_xlabel('Iteration')
ax3.set_xlabel('Iteration')
ax4.set_xlabel('Iteration')

figure.suptitle("Newton's Method for: \n $x^2 - 500x + y^2 - 30y + 5z^2 - 60z$", fontsize=16)

ax1.set_title('$x_1$')
ax2.set_title('$x_2$')
ax3.set_title('$x_3$')
ax4.set_title("Function value")

plt.show()
