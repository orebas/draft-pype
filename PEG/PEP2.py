import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


class ParameterEstimationProblem:
    def __init__(self, ode_system, true_params, initial_conditions, observed_variables):
        self.ode_system = ode_system
        self.true_params = true_params
        self.initial_conditions = initial_conditions
        self.observed_variables = observed_variables
        self.compile_ode_func()

    def compile_ode_func(self):
        param_symbols = list(self.ode_system.params.values())
        state_symbols = list(self.ode_system.state_vars.values())

        ode_exprs = [
            self.ode_system.equations[var] for var in self.ode_system.state_vars
        ]
        self.ode_func_lambda = sp.lambdify(
            [self.ode_system.t] + state_symbols + param_symbols, ode_exprs
        )

    def vectorized_ode_func(self, t, y, *params):
        return self.ode_func_lambda(t, *y, *params)

    def generate_data(
        self,
        time_points,
        method="DOP853",
        inject_noise=False,
        noise_params=None,
        params=None,
        initial_conditions=None,
        **solver_kwargs,
    ):
        if params is None:
            params = self.true_params
        if initial_conditions is None:
            initial_conditions = self.initial_conditions

        y0 = [initial_conditions[var] for var in self.ode_system.state_vars]
        param_values = [params[p] for p in self.ode_system.params]

        try:
            solution = solve_ivp(
                lambda t, y: self.vectorized_ode_func(t, y, *param_values),
                (time_points[0], time_points[-1]),
                y0,
                t_eval=time_points,
                method=method,
                vectorized=False,
                **solver_kwargs,
            )
        except Exception as e:
            print(f"ODE solution failed: {str(e)}")
            return None

        if not solution.success:
            print(f"ODE solution failed: {solution.message}")
            return None

        data_sample = {"t": time_points}
        for var, values in zip(self.ode_system.state_vars.keys(), solution.y):
            data = np.array(values)
            if inject_noise and noise_params:
                noise = np.random.normal(
                    noise_params["mean"], noise_params["std"], size=data.shape
                )
                data += noise
            data_sample[var] = data

        for name, expr in self.ode_system.measured_quantities.items():
            param_dict = {sp.Symbol(k): sp.Float(v) for k, v in params.items()}
            data = np.array(
                [
                    float(
                        expr.subs(param_dict).subs(
                            {
                                self.ode_system.state_vars[var]: sp.Float(
                                    data_sample[var][i]
                                )
                                for var in self.ode_system.state_vars
                            }
                        )
                    )
                    for i in range(len(time_points))
                ]
            )
            if inject_noise and noise_params:
                noise = np.random.normal(
                    noise_params["mean"], noise_params["std"], size=data.shape
                )
                data += noise
            data_sample[name] = data

        return data_sample

    def estimate_parameters(self, observed_data, bounds=None, num_starts=10):
        if observed_data is None or "t" not in observed_data:
            raise ValueError("Invalid observed_data. Make sure it contains 't' key.")

        param_names = list(self.ode_system.params.keys())
        state_var_names = list(self.ode_system.state_vars.keys())

        def callback(xk):
            print(
                f"Iteration: {callback.iteration}, Objective: {objective_function(xk):.6f}"
            )
            callback.iteration += 1

        if bounds is None:
            bounds = [(0, 5) for _ in range(len(param_names) + len(state_var_names))]

        def objective_function(x):
            current_params = dict(zip(param_names, x[: len(param_names)]))
            current_initial_conditions = dict(
                zip(state_var_names, x[len(param_names) :])
            )

            simulated_data = self.generate_data(
                observed_data["t"],
                params=current_params,
                initial_conditions=current_initial_conditions,
            )

            if simulated_data is None:
                return 1e10  # Return a large penalty value for failed ODE solutions

            error = 0
            for var in state_var_names:
                error += np.sum((observed_data[var] - simulated_data[var]) ** 2)
            return error

        best_result = None
        best_error = np.inf

        for start in range(num_starts):
            initial_guess = np.random.uniform(
                low=[b[0] for b in bounds], high=[b[1] for b in bounds]
            )
            callback.iteration = 0
            print(f"\nStart {start + 1}/{num_starts}")

            result = minimize(
                objective_function,
                initial_guess,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-8, "maxiter": 1000},
                callback=callback,
            )

            print(f"Final objective: {result.fun:.6f}")

            if result.success and result.fun < best_error:
                best_result = result
                best_error = result.fun

        if best_result is not None:
            estimated_params = dict(zip(param_names, best_result.x[: len(param_names)]))
            estimated_initial_conditions = dict(
                zip(state_var_names, best_result.x[len(param_names) :])
            )
            return estimated_params, estimated_initial_conditions
        else:
            raise RuntimeError(
                "Parameter estimation failed: No successful optimizations"
            )

    def calculate_sensitivities(
        self, time_points, params=None, initial_conditions=None
    ):
        if params is None:
            params = self.true_params
        if initial_conditions is None:
            initial_conditions = self.initial_conditions

        param_names = list(self.ode_system.params.keys())
        state_var_names = list(self.ode_system.state_vars.keys())

        n_states = len(state_var_names)
        n_params = len(param_names)
        n_obs = len(self.observed_variables)
        n_times = len(time_points)

        # Create symbolic variables for states and parameters
        state_symbols = [self.ode_system.state_vars[name] for name in state_var_names]
        param_symbols = [self.ode_system.params[name] for name in param_names]
        t_symbol = sp.Symbol("t")

        # Create symbolic expressions for the ODEs
        ode_exprs = [self.ode_system.equations[name] for name in state_var_names]

        # Create symbolic expressions for the observed variables
        obs_exprs = [
            self.ode_system.measured_quantities[name]
            for name in self.observed_variables
        ]

        # Create lambdified functions for the ODEs and observed variables
        ode_funcs = [
            sp.lambdify([t_symbol] + state_symbols + param_symbols, expr)
            for expr in ode_exprs
        ]
        obs_funcs = [
            sp.lambdify([t_symbol] + state_symbols + param_symbols, expr)
            for expr in obs_exprs
        ]

        # Create sensitivity expressions for state variables
        state_sensitivity_exprs = []
        for i, state in enumerate(state_var_names):
            for j, param in enumerate(param_names):
                expr = sum(
                    ode_exprs[i].diff(state_symbols[k]) * sp.Symbol(f"s_{k}_{j}")
                    for k in range(n_states)
                )
                expr += ode_exprs[i].diff(param_symbols[j])
                state_sensitivity_exprs.append(expr)

        # Create sensitivity expressions for observed variables
        obs_sensitivity_exprs = []
        for i, obs in enumerate(self.observed_variables):
            for j, param in enumerate(param_names):
                expr = sum(
                    obs_exprs[i].diff(state_symbols[k]) * sp.Symbol(f"s_{k}_{j}")
                    for k in range(n_states)
                )
                expr += obs_exprs[i].diff(param_symbols[j])
                obs_sensitivity_exprs.append(expr)

        # Create lambdified functions for the sensitivities
        state_sensitivity_funcs = [
            sp.lambdify(
                [t_symbol]
                + state_symbols
                + param_symbols
                + [
                    sp.Symbol(f"s_{i}_{j}")
                    for i in range(n_states)
                    for j in range(n_params)
                ],
                expr,
            )
            for expr in state_sensitivity_exprs
        ]

        obs_sensitivity_funcs = [
            sp.lambdify(
                [t_symbol]
                + state_symbols
                + param_symbols
                + [
                    sp.Symbol(f"s_{i}_{j}")
                    for i in range(n_states)
                    for j in range(n_params)
                ],
                expr,
            )
            for expr in obs_sensitivity_exprs
        ]

        # Define the combined ODE system for states and sensitivities
        def combined_ode(t, y):
            states = y[:n_states]
            sensitivities = y[n_states:].reshape(n_states, n_params)

            dydt = [f(t, *states, *params.values()) for f in ode_funcs]
            dsdt = [
                f(t, *states, *params.values(), *sensitivities.flatten())
                for f in state_sensitivity_funcs
            ]

            return np.array(dydt + dsdt)

        # Initial conditions for states and sensitivities
        y0 = list(initial_conditions.values()) + [0] * (n_states * n_params)

        # Solve the combined ODE system
        solution = solve_ivp(
            combined_ode,
            (time_points[0], time_points[-1]),
            y0,
            t_eval=time_points,
            method="LSODA",
        )

        if not solution.success:
            raise ValueError(f"ODE solution failed: {solution.message}")

        # Extract state sensitivities from the solution
        S_states = (
            solution.y[n_states:]
            .reshape(n_states, n_params, n_times)
            .transpose(2, 0, 1)
        )

        # Calculate observed variable sensitivities
        S_obs = np.zeros((n_times, n_obs, n_params))
        for i in range(n_times):
            states = solution.y[:n_states, i]
            state_sens = S_states[i].T.flatten()
            for j, f in enumerate(obs_sensitivity_funcs):
                S_obs[i, j // n_params, j % n_params] = f(
                    time_points[i], *states, *params.values(), *state_sens
                )

        return S_obs

    def plot_sensitivities(self, S, time_points):
        import matplotlib.pyplot as plt

        param_names = list(self.ode_system.params.keys())
        n_obs = len(self.observed_variables)
        n_params = len(param_names)

        fig, axes = plt.subplots(
            n_obs, n_params, figsize=(4 * n_params, 3 * n_obs), sharex=True
        )

        for i, obs in enumerate(self.observed_variables):
            for j, param in enumerate(param_names):
                ax = (
                    axes[i, j]
                    if n_obs > 1 and n_params > 1
                    else axes[j] if n_obs == 1 else axes[i]
                )
                ax.plot(time_points, S[:, i, j])
                ax.set_title(f"{obs} sensitivity to {param}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Sensitivity")

        plt.tight_layout()
        plt.show()

    def symbolic_sensitivity_analysis(self, max_derivatives=3):
        param_names = list(self.ode_system.params.keys())
        state_var_names = list(self.ode_system.state_vars.keys())
        obs_var_names = list(self.observed_variables)

        # Create symbolic variables for states, parameters, and time
        state_symbols = [self.ode_system.state_vars[name] for name in state_var_names]
        param_symbols = [self.ode_system.params[name] for name in param_names]
        t_symbol = sp.Symbol("t")

        # Create symbolic expressions for the ODEs
        ode_exprs = [self.ode_system.equations[name] for name in state_var_names]

        # Create symbolic expressions for the observed variables
        obs_exprs = [
            self.ode_system.measured_quantities[name] for name in obs_var_names
        ]

        # Create a dictionary to store the derivatives of observed variables
        obs_derivatives = {name: [expr] for name, expr in zip(obs_var_names, obs_exprs)}

        # Calculate higher-order derivatives of observed variables
        for _ in range(1, max_derivatives + 1):
            for name in obs_var_names:
                prev_deriv = obs_derivatives[name][-1]
                next_deriv = sum(
                    prev_deriv.diff(state) * ode
                    for state, ode in zip(state_symbols, ode_exprs)
                )
                obs_derivatives[name].append(next_deriv)

        # Create the observability matrix
        obs_matrix_rows = []
        for name in obs_var_names:
            obs_matrix_rows.extend(
                [
                    deriv.diff(param)
                    for deriv in obs_derivatives[name]
                    for param in param_symbols
                ]
            )

        obs_matrix = Matrix(obs_matrix_rows)

        # Calculate the rank of the observability matrix
        rank = obs_matrix.rank()

        # Determine which parameters are identifiable
        identifiable_params = []
        derivatives_needed = {}
        for i, param in enumerate(param_names):
            param_column = obs_matrix.col(i)
            if any(entry != 0 for entry in param_column):
                identifiable_params.append(param)
                # Find the minimum number of derivatives needed for this parameter
                for j, entry in enumerate(param_column):
                    if entry != 0:
                        obs_var = obs_var_names[j // (max_derivatives + 1)]
                        deriv_order = j % (max_derivatives + 1)
                        if (
                            param not in derivatives_needed
                            or derivatives_needed[param][1] > deriv_order
                        ):
                            derivatives_needed[param] = (obs_var, deriv_order)

        return {
            "rank": rank,
            "total_parameters": len(param_names),
            "identifiable_parameters": identifiable_params,
            "derivatives_needed": derivatives_needed,
            "observability_matrix": obs_matrix,
        }

    def print_identifiability_analysis(self, result):
        print(f"Rank of observability matrix: {result['rank']}")
        print(f"Total number of parameters: {result['total_parameters']}")
        print(
            f"Number of identifiable parameters: {len(result['identifiable_parameters'])}"
        )
        print("\nIdentifiable parameters:")
        for param in result["identifiable_parameters"]:
            obs_var, deriv_order = result["derivatives_needed"][param]
            print(
                f"  {param}: requires {deriv_order}{'st' if deriv_order == 1 else 'nd' if deriv_order == 2 else 'th'} derivative of {obs_var}"
            )

        print("\nNon-identifiable parameters:")
        non_identifiable = set(self.ode_system.params.keys()) - set(
            result["identifiable_parameters"]
        )
        for param in non_identifiable:
            print(f"  {param}")

    def local_identifiability_analysis(problem):
        pass
        # Function to perform local identifiability analysis

    def construct_equation_system(problem):
        pass
        # Function to construct the equation system for solving

    def solve_ode_system(problem, params, initial_conditions, time_points):
        pass
        # Function to solve the ODE system
