import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


class ODESystem:
    def __init__(self):
        self.t = sp.Symbol("t")
        self.state_vars = {}
        self.params = {}
        self.equations = {}
        self.measured_quantities = {}

    def add_state_variable(self, name):
        self.state_vars[name] = sp.Function(name)(self.t)
        return self.state_vars[name]

    def add_parameter(self, name):
        self.params[name] = sp.Symbol(name)
        return self.params[name]

    def add_equation(self, var, expr):
        if var not in self.state_vars:
            raise ValueError(f"State variable '{var}' not found in the system.")
        self.equations[var] = expr

    def add_measured_quantity(self, name, expr):
        used_symbols = expr.free_symbols
        valid_symbols = (
            set(self.state_vars.values()) | set(self.params.values()) | {self.t}
        )
        if not used_symbols.issubset(valid_symbols):
            invalid_symbols = used_symbols - valid_symbols
            raise ValueError(f"Invalid symbols in expression: {invalid_symbols}")
        self.measured_quantities[name] = expr

    def get_derivative(self, var, order=1):
        if var not in self.state_vars:
            raise ValueError(f"State variable '{var}' not found in the system.")
        expr = self.equations[var]
        for _ in range(order):
            expr = expr.diff(self.t)
        return expr
