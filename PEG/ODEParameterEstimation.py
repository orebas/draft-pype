import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class ParameterEstimationResult:
    parameters: Dict[str, float]
    states: Dict[str, float]
    at_time: float
    err: float
    return_code: int
    datasize: int
    report_time: float


def parameter_estimation(
    model_func: Callable, t_span, y0, params_guess, data, measured_vars
):
    """
    Estimate parameters of an ODE model by minimizing the error between the model and data.

    Arguments:
    - model_func: function defining the ODE system
    - t_span: tuple, time interval
    - y0: initial conditions
    - params_guess: initial guess for parameters
    - data: observed data
    - measured_vars: indices of variables that are measured

    Returns:
    - ParameterEstimationResult
    """

    def objective(params):
        sol = solve_ivp(
            lambda t, y: model_func(t, y, params), t_span, y0, t_eval=data["t"]
        )
        error = 0.0
        for idx, var in enumerate(measured_vars):
            error += np.sum((sol.y[var] - data["y"][idx]) ** 2)
        return error

    result = minimize(objective, params_guess)
    estimated_params = result.x
    return ParameterEstimationResult(
        parameters=dict(
            zip(
                ["param{}".format(i) for i in range(len(estimated_params))],
                estimated_params,
            )
        ),
        states={},  # Can be filled with final state values
        at_time=t_span[1],
        err=result.fun,
        return_code=result.status,
        datasize=len(data["t"]),
        report_time=t_span[1],
    )


# Example model function
def model_func(t, y, params):
    # Example ODE: dy/dt = -params[0] * y
    return [-params[0] * y[0]]


# Example usage
t_span = (0, 10)
y0 = [1.0]
params_guess = [0.1]
data = {"t": np.linspace(0, 10, 100), "y": [np.exp(-0.5 * np.linspace(0, 10, 100))]}
measured_vars = [0]
result = parameter_estimation(model_func, t_span, y0, params_guess, data, measured_vars)
print(result)
