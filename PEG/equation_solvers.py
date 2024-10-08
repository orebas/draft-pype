import numpy as np
from sympy import symbols, Eq, solve as sympy_solve
from scipy.optimize import fsolve


def solve_system_with_fsolve(equations, variables, initial_guess):
    """
    Solve a system of equations numerically using fsolve.

    Arguments:
    - equations: list of sympy expressions
    - variables: list of sympy symbols
    - initial_guess: initial guess for the variables

    Returns:
    - Solution as a dictionary
    """
    func = lambda x: [eq.subs(dict(zip(variables, x))).evalf() for eq in equations]
    sol = fsolve(func, initial_guess)
    return dict(zip(variables, sol))


# Example usage:
x, y = symbols("x y")
equations = [Eq(x**2 + y**2, 1), Eq(x - y, 0)]
variables = [x, y]
initial_guess = [0.5, 0.5]
solution = solve_system_with_fsolve(
    [eq.lhs - eq.rhs for eq in equations], variables, initial_guess
)
print(solution)
