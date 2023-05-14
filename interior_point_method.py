import numpy as np 
import numdifftools as nd 
import matplotlib.pyplot as plt
import sympy as sym
import sys



# Starting values, must be feasible. If a constraint says x_1 > 0, you must start with an initial value x_1 > 0 (hence the interior point method name)
xi = np.array([
    [1.2],
    [1.3],
    [1.1]
])

# Note: will use the name interior point method and barrier method as the same thing.

def calculateHessianAndGradient(xi):

    #Defining 3 variables
    x1, x2, x3 = sym.symbols('x1 x2 x3')

    # Defining the objective function we want to minimize, with barrier method parameter t
    # This example, arbitrary function with 3 constraints: "x1 > 0", "x2 > 0" and "x3 > 0"
    function = t*(x1 + x2 + x3) - (sym.log(-(x1**2+x2**2+x3**2-10)))
    
    # Another function example below (uncomment it)
    # Note this function could benefit with a bigger starting value t if you have not tweaked with the starting values
    #function = (x1*sym.log(x1)+x2*sym.log(x2)+x3*sym.log(x3))-(1/t)*(sym.log(-(-x1)) + sym.log(-(-x2)) + sym.log(-(-x3)) )

    # Note also writing it like "t" and "(1/t)" in the two examples are the same thing

    #function value at xi
    function_value = function.evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})

    # Check if the answer is feasible. A too big t value can cause numerical difficulties 
    # If this happens, maybe start with lower t value, decrease the t multiplier, possibly decrease epsilon or increasee the tolerance
    if sym.im(function_value) != 0:
        
        print('=========COMPLEX ANSWER, NOT FEASIBLE =============') 
        print("f(x)= ", function_value)
        print("t value: ", t)
        sys.exit()
    
    # Derivating f(x) for x1, x2, x3 (algebraic answer, without values)
    der_x1 = function.diff(x1)
    der_x2 = function.diff(x2)
    der_x3 = function.diff(x3)

    # Putting values into the derivatives
    der_x1_values = function.diff(x1).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x2_values = function.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x3_values = function.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})

    # Putting the derivatives together in a matrix so we get the gradient of the objective function
    gradient_values = np.array([
        [der_x1_values],
        [der_x2_values],
        [der_x3_values]
    ], dtype=np.float32)

    # Derivating the objective function further to get the hessian
    der_x1x1_values = der_x1.diff(x1).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_crossx1x2_values = der_x1.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x2x2_values = der_x2.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_crossx1x3_values = der_x1.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x3x3_values = der_x3.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_crossx2x3_values = der_x2.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})

    # Putting the derivatives together in a 3x3 matrix so we get the hessian of the objective function
    hessian_values = np.array([
        [der_x1x1_values, der_crossx1x2_values, der_crossx1x3_values],
        [der_crossx1x2_values, der_x2x2_values, der_crossx2x3_values],
        [der_crossx1x3_values, der_crossx2x3_values, der_x3x3_values]
    ], dtype=np.float32)
    
    return gradient_values, hessian_values

# For plotting
x1_list = np.array(xi[0])
x2_list = np.array(xi[1]) 
x3_list = np.array(xi[2]) 
NR_iteration = 1 

# Used for the stopping criterion in the Newton's method
lambda_2 = 1

# Initialize the number of outer loops
i = 1 

# if statement later
c = 1E-6 

# number of constraints
m = 1  

# How exact one wants the end result to be. lower tolerance => more exact. (A too low value could as a result increase "t" too much and cause issues).
tolerance = 0.002

# Step size. How big of a step newton's method takes. Should not be set over 1.
epsilon = 0.5

# t multiplier
mu = 2

# Starting value t. Note this can start larger if the intial guess is closer to the optimal point. A too big starting t => a bad time 
t = 0.1

# Outer loop
while (m/t) > tolerance:
    print("main loop i: ", i)

    # So the code goes in the inner loop after t is changed 
    lambda_2 = 1

    # Reset temporary inner loop number 
    j = 1 

    # Inner loop (Newton's method on function by given t) 
    while (lambda_2/2) > tolerance:

        print("newton temporary loop j:", j)

        gradient, hessian = calculateHessianAndGradient(xi)

        # The Newton decrement squared = lamba**2 = lambda_2, used for the stopping criterion
        lambda_2 = np.transpose(gradient)@np.linalg.inv(hessian)@gradient  

        # In order to increase the basin of attraction and the robustness of the algorithm, it is suitable to force: hessian >= c*I
        if (np.abs(np.linalg.eigvals(hessian)) < c).all():
                hessian = c*np.identity(3) 
 
        # Newton's method
        xi = (1-epsilon)*xi+epsilon*(np.linalg.inv(hessian))@(hessian@xi-gradient)

        # Putting the results of the Newton's method into a list so we can plot it later
        x1_list = np.append(x1_list, xi[0])
        x2_list = np.append(x2_list, xi[1])
        x3_list = np.append(x3_list, xi[2])

        NR_iteration += 1
        j += 1

        # Useful for debugging
        print("h:\n", hessian)
        print("g: \n", gradient)
        print("xi: \n", xi)
        print("---")
    print("=====DONE NR=====")

    # Increasing t 
    t = 2*t
    print("t :", t)

    # Number of outer loops increases by 1.
    i+=1
    
print("====DONE OUTER LOOP!====")   
    
# Plotting stuff 
figure, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14, 8))

ax1.plot(np.arange(NR_iteration), x1_list, "g")
ax2.plot(np.arange(NR_iteration), x2_list, "r")
ax3.plot(np.arange(NR_iteration), x3_list, "k")

ax1.grid()
ax2.grid()
ax3.grid()

ax1.set_xlabel('iterations')
ax2.set_xlabel('iterations')
ax3.set_xlabel('iterations')

ax1.set_title('$x_1$')
ax2.set_title('$x_2$')
ax3.set_title('$x_3$')

figure.suptitle("Interior method f(x1, x2, x3)", fontsize=16)

plt.show()
