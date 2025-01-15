import numpy as np
import matplotlib.pyplot as plt


def CDopt(f, dim, n_particles, max_iter, bounds):
    coords = np.zeros((max_iter, dim))

    lb, ub = bounds
    sa, sfa, a_score = 0.16e5, 0.25, float('inf')
    sb, sfb, b_score = 2.7e5, 0.5, float('inf')
    sg, sfg, g_score = 3e5, 1.0, float('inf')

    # Initialize positions
    pos = lb + (ub - lb) * np.random.rand(n_particles, dim)

    for iter in range(max_iter):
        pos = np.clip(pos, lb, ub)

        for particle_idx in range(n_particles):
            fit = f(pos[particle_idx, :])
            if fit < a_score:
                a_score, a_pos = fit, pos[particle_idx, :].copy()
            elif fit < b_score:
                b_score, b_pos = fit, pos[particle_idx, :].copy()
            elif fit < g_score:
                g_score, g_pos = fit, pos[particle_idx, :].copy()

        WSh = 3 - 3 * (iter / max_iter)  # Decreasing walking speed factor

        for particle_idx in range(n_particles):
            v = gP_vectorized(pos[particle_idx, :], a_pos, sa, sfa, WSh)
            v += gP_vectorized(pos[particle_idx, :], b_pos, sb, sfb, WSh)
            v += gP_vectorized(pos[particle_idx, :], g_pos, sg, sfg, WSh)
            pos[particle_idx, :] = v / 3

        coords[iter, :] = a_pos

    xmin = a_pos
    fmin = a_score
    return xmin, fmin, coords


def gP_vectorized(pC, rC, s, sf, WSh):
    xh = (np.random.rand(len(pC)) ** 2) * np.pi  # Random human area factor
    S = np.log10(1 + (s - 1) * np.random.rand(len(pC)))
    p = xh / (sf * S) - (WSh * np.random.rand(len(pC)))
    A = (np.random.rand(len(pC)) ** 2) * np.pi  # Random particle area factor
    D = np.abs(A * rC - pC)
    v = sf * (rC - p * D)
    return v


def testSuit(function_name):
    functions = {
        # Many local minima
        "Ackley": (lambda x: -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))) -
                             np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20,
                   2, [-5, 5], [0, 0], 0),
        "Eggholder": (lambda x: -(x[1] + 47) * np.sin(np.sqrt(abs(x[1] + x[0] / 2 + 47))) - x[0] * np.sin(
            np.sqrt(abs(x[0] - (x[1] + 47)))),
                      2, [-512, 512], [512, 404.2319], -959.6407),
        # Bowl-shaped
        "Sphere": (lambda x: sum(xi ** 2 for xi in x), 2, [-5.12, 5.12], [0, 0], 0),
        "Trid": (lambda x: sum((xi - 1) ** 2 for xi in x) - sum(x[i] * x[i - 1] for i in range(1, len(x))),
                 2, [-4, 4], [2, 2], -2),
        # Plate-shaped
        "McCormick": (lambda x: np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1,
                      2, [-1.5, 4], [-0.54719, -1.54719], -1.9133),
        "Booth": (lambda x: (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2,
                  2, [-10, 10], [1, 3], 0),
        # Valley-shaped
        "Rosenbrock": (lambda x: sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)),
                       2, [-2.048, 2.048], [1, 1], 0),
        "SixHumpCamel": (lambda x: (4 - 2.1 * x[0] ** 2 + (x[0] ** 4) / 3) * (x[0] ** 2) + x[0] * x[1] +
                                   (-4 + 4 * x[1] ** 2) * (x[1] ** 2),
                         2, [-2, 2], [0.0898, -0.7126], -1.0316),
        # Steep ridges/drops
        "Easom": (lambda x: -np.cos(x[0])*np.cos(x[1])*np.exp(-(x[0]-np.pi)**2-(x[1]-np.pi)**2),
                  2, [0, 2*np.pi], [np.pi, np.pi], -1),
        "Michalewicz": (lambda x: -1 * sum(np.sin(x[i]) * (np.sin(((i+1) * x[i] ** 2) / np.pi)) ** 2 for i in range(2)),
                        2, [0, np.pi], [2.20, 1.57], -1.8013),

    }
    return functions[function_name]


def plot_trajectory(f, bounds, coords):
    x = np.linspace(bounds[0], bounds[1], 500)
    y = np.linspace(bounds[0], bounds[1], 500)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f([xi, yi]) for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

    plt.figure()
    plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title('Optimization Trajectory')

    plt.plot(coords[:, 0], coords[:, 1], 'r-o', label='Trajectory')
    plt.plot(coords[-1, 0], coords[-1, 1], 'g*', label='Final Point', markersize=10)
    plt.legend()
    plt.show()


def main():
    function_name = "Michalewicz"  # Change to desired function
    f, dim, bounds, xmin_true, fmin_true = testSuit(function_name)

    n_particles = 30
    max_iter = 100

    xmin, fmin, coords = CDopt(f, dim, n_particles, max_iter, bounds)

    print(f"Test function: {function_name}")
    print(f"True minimum: x = {xmin_true}, f(x) = {fmin_true}")
    print(f"Obtained minimum: x = {xmin}, f(x) = {fmin}")
    print(f"Error in x: {np.linalg.norm(np.array(xmin) - np.array(xmin_true))}")
    print(f"Error in f(x): {abs(fmin - fmin_true)}")

    plot_trajectory(f, bounds, coords)


if __name__ == "__main__":
    main()
