import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from ode_system import ODESystem, create_ode_system
from parameter_estimation_problem import ParameterEstimationProblem


def test_simple_oscillator():
    # Define the system
    t = sp.Symbol("t")
    x1, x2 = sp.Function("x1")(t), sp.Function("x2")(t)
    a, b = sp.symbols("a b")

    D = sp.Function("D")
    equations = [
        sp.Eq(sp.Derivative(x1, t), -a * x2),
        sp.Eq(sp.Derivative(x2, t), b * x1),
    ]

    model, _ = create_ode_system("simple_oscillator", [x1, x2], [a, b], equations, t)

    # Define measured quantities
    y1, y2 = sp.Function("y1")(t), sp.Function("y2")(t)
    measured_quantities = [sp.Eq(y1, x1), sp.Eq(y2, x2)]

    # Create a ParameterEstimationProblem
    pep = ParameterEstimationProblem(
        name="simple_oscillator",
        model=model,
        measured_quantities=measured_quantities,
        p_true=[0.4, 0.8],
        ic=[1.0, 0.0],
        unident_count=0,
        time_interval=[0, 10],
        datasize=100,
    )

    # Print problem details
    print(pep)

    # Plot the generated data
    pep.plot_data()

    # Test serialization and deserialization
    pep_dict = pep.to_dict()
    pep_recovered = ParameterEstimationProblem.from_dict(pep_dict)
    print("\nRecovered problem:")
    print(pep_recovered)

    # Compare original and recovered data
    for key in pep.data_sample:
        assert np.allclose(
            pep.data_sample[key], pep_recovered.data_sample[key]
        ), f"Mismatch in {key} data"

    print("\nSerialization and deserialization test passed.")


if __name__ == "__main__":
    test_simple_oscillator()
