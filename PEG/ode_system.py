import sympy as sp
from sympy.core.symbol import Symbol
from sympy.core.function import Function
from typing import List, Dict, Tuple

class ODESystem:
    def __init__(self, name: str, equations: List[sp.Equality], states: List[Function], 
                 parameters: List[Symbol], time_variable: Symbol):
        self.name = name
        self.equations = equations
        self.states = states
        self.parameters = parameters
        self.time_variable = time_variable
        self.measured_quantities: List[sp.Equality] = []

    @property
    def state_variables(self) -> List[Symbol]:
        return [state.func for state in self.states]

    @property
    def parameter_variables(self) -> List[Symbol]:
        return self.parameters

    def add_measured_quantity(self, equation: sp.Equality):
        self.measured_quantities.append(equation)

    def get_rhs(self) -> List[sp.Expr]:
        return [eq.rhs for eq in self.equations]

    def get_lhs(self) -> List[sp.Expr]:
        return [eq.lhs for eq in self.equations]

    def subs(self, subs_dict: Dict[Symbol, float]) -> 'ODESystem':
        new_equations = [eq.subs(subs_dict) for eq in self.equations]
        new_states = [state.subs(subs_dict) for state in self.states]
        new_parameters = [param for param in self.parameters if param not in subs_dict]
        new_measured_quantities = [eq.subs(subs_dict) for eq in self.measured_quantities]

        new_system = ODESystem(self.name, new_equations, new_states, new_parameters, self.time_variable)
        new_system.measured_quantities = new_measured_quantities
        return new_system

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'equations': self.equations,
            'states': self.states,
            'parameters': self.parameters,
            'time_variable': self.time_variable,
            'measured_quantities': self.measured_quantities
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ODESystem':
        system = cls(data['name'], data['equations'], data['states'], data['parameters'], data['time_variable'])
        system.measured_quantities = data['measured_quantities']
        return system

    def __repr__(self):
        return f"ODESystem(name={self.name}, states={self.state_variables}, parameters={self.parameters})"

def create_ode_system(name: str, states: List[Function], parameters: List[Symbol], 
                      equations: List[sp.Equality], time_variable: Symbol) -> Tuple[ODESystem, List[sp.Equality]]:
    system = ODESystem(name, equations, states, parameters, time_variable)
    return system, []  # The second element is for measured quantities, initially empty

# Example usage:
if __name__ == "__main__":
    t = sp.Symbol('t')
    x1, x2 = sp.Function('x1')(t), sp.Function('x2')(t)
    a, b = sp.symbols('a b')
    
    D = sp.Function('D')
    
    equations = [
        sp.Eq(D(x1), -a * x2),
        sp.Eq(D(x2), b * x1)
    ]
    
    model, _ = create_ode_system("simple", [x1, x2], [a, b], equations, t)
    
    print(model)
    print("Equations:", model.equations)
    print("States:", model.state_variables)
    print("Parameters:", model.parameter_variables)
    
    # Adding measured quantities
    y1, y2 = sp.Function('y1')(t), sp.Function('y2')(t)
    model.add_measured_quantity(sp.Eq(y1, x1))
    model.add_measured_quantity(sp.Eq(y2, x2))
    
    print("Measured Quantities:", model.measured_quantities)
    
    # Substituting values
    subs_model = model.subs({a: 0.5})
    print("Substituted model:", subs_model)
    print("Substituted equations:", subs_model.equations)