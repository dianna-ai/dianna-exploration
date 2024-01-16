import math
import matplotlib.pyplot as plt
import numpy as np

thetas = np.linspace(-2 * np.pi / 3, 4 * np.pi / 3, 300)


def ang(thetas):
    new_thetas = np.zeros_like(thetas)
    new_thetas[np.logical_and(0 < thetas, thetas < 2 * np.pi / 3)] = thetas[np.logical_and(0 < thetas, thetas < 2 * np.pi / 3)] + 2 * np.pi / 3
    new_thetas[np.logical_and(2 * np.pi / 3 < thetas, thetas < 4 * np.pi / 3)] = thetas[np.logical_and(2 * np.pi / 3 < thetas, thetas < 4 * np.pi / 3)]
    new_thetas[np.logical_and(-4 * np.pi / 3 < thetas, thetas < 0)] = thetas[np.logical_and(-4 * np.pi / 3 < thetas, thetas < 0)] - 2 * np.pi / 3
    return new_thetas


def polar_triangle(thetas):
    return -1.0 / (2.0 * np.cos(ang(thetas)))


def spherical_triangle(thetas, n=1):
    return polar_triangle(thetas) ** (1.0 / n)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


nus = np.linspace(1.0, 100.0, 36) ** 0.2
# Plot
n_rows = 6
n_cols = math.ceil(len(nus) / n_rows)
fig, ax = plt.subplots(n_rows, n_cols)
for i, nu in enumerate(nus):
    row_idx = i // n_cols
    col_idx = i % n_cols
    rhos = spherical_triangle(thetas, n=nu)
    x, y = pol2cart(rhos, thetas)
    ax[row_idx, col_idx].plot(x, y, color='k', linewidth=10)
    ax[row_idx, col_idx].set_aspect('equal')
    ax[row_idx, col_idx].set_axis_off()

plt.show()

