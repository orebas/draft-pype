def sample_data(model_func, t_span, y0, params, num_points=100, noise_std=0.0):
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    sol = solve_ivp(lambda t, y: model_func(t, y, params), t_span, y0, t_eval=t_eval)
    data = {'t': sol.t, 'y': sol.y}
    if noise_std > 0:
        data['y'] += np.random.normal(0, noise_std, size=sol.y.shape)
    return data
