import numpy as np
import sympy as sp
from typing import Dict, List, Union, Callable, Optional
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field

from ode_system import ODESystem


@dataclass
class ParameterEstimationProblem:
    name: str
    model: ODESystem
    measured_quantities: List[sp.Equality]
    data_sample: Optional[Dict[str, np.ndarray]] = None
    solver: Optional[Callable] = None
    p_true: List[float] = field(default_factory=list)
    ic: List[float] = field(default_factory=list)
    unident_count: int = 0
    time_interval: List[float] = field(default_factory=lambda: [-0.5, 0.5])
    datasize: int = 21

    def __post_init__(self):
        if self.data_sample is None:
            self.generate_sample_data()

    def generate_sample_data(self):
        """Generate sample data for the problem."""
        if not self.solver:
            self.solver = solve_ivp

        t_eval = np.linspace(
            self.time_interval[0], self.time_interval[1], self.datasize
        )

        # Create lambda functions for each equation
        ode_lambdas = []
        for eq in self.model.equations:
            rhs = eq.rhs
            for state in self.model.states:
                rhs = rhs.subs(
                    sp.Derivative(state, self.model.time_variable),
                    sp.Function(f"d{state.func.__name__}")(self.model.time_variable),
                )
            lamb = sp.lambdify(
                [self.model.time_variable]
                + [state.func for state in self.model.states]
                + self.model.parameters,
                rhs,
                modules=["numpy"],
            )
            ode_lambdas.append(lamb)

        def ode_func(t, y, *p):
            return [f(t, *y, *p) for f in ode_lambdas]

        solution = self.solver(
            ode_func,
            [self.time_interval[0], self.time_interval[1]],
            self.ic,
            args=tuple(self.p_true),
            t_eval=t_eval,
        )

        self.data_sample = {"t": solution.t}

        # Create lambda functions for measured quantities
        mq_lambdas = []
        for mq in self.measured_quantities:
            lamb = sp.lambdify(
                [self.model.time_variable]
                + [state.func for state in self.model.states],
                mq.rhs,
                modules=["numpy"],
            )
            mq_lambdas.append(lamb)

        for i, mq_lambda in enumerate(mq_lambdas):
            y = mq_lambda(solution.t, *solution.y)
            self.data_sample[f"y{i+1}"] = np.array(y)

    def plot_data(self):
        """Plot the generated or provided data."""
        import matplotlib.pyplot as plt

        if self.data_sample is None:
            raise ValueError("No data sample available. Generate data first.")

        t = self.data_sample["t"]
        fig, axs = plt.subplots(
            len(self.measured_quantities),
            1,
            figsize=(10, 4 * len(self.measured_quantities)),
        )

        if len(self.measured_quantities) == 1:
            axs = [axs]

        for i, (ax, mq) in enumerate(zip(axs, self.measured_quantities)):
            y = self.data_sample[f"y{i+1}"]
            ax.plot(t, y, "o-")
            ax.set_title(f"Measured Quantity: {mq.lhs}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")

        plt.tight_layout()
        plt.show()

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "model": self.model.to_dict(),
            "measured_quantities": self.measured_quantities,
            "data_sample": self.data_sample,
            "p_true": self.p_true,
            "ic": self.ic,
            "unident_count": self.unident_count,
            "time_interval": self.time_interval,
            "datasize": self.datasize,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ParameterEstimationProblem":
        data["model"] = ODESystem.from_dict(data["model"])
        return cls(**data)

    def __repr__(self):
        return (
            f"ParameterEstimationProblem(name={self.name}, "
            f"model={self.model}, "
            f"p_true={self.p_true}, "
            f"ic={self.ic}, "
            f"unident_count={self.unident_count})"
        )
