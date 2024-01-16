"""
Inspired by https://math.stackexchange.com/questions/786335/is-there-a-function-that-draws-a-triangle-with-rounded-edges
"""
import math
import matplotlib.pyplot as plt
import numpy as np


def ang(thetas):
    new_thetas = np.zeros_like(thetas)
    new_thetas[np.logical_and(0 <= thetas, thetas < 2 * np.pi / 3)] = thetas[np.logical_and(0 <= thetas, thetas < 2 * np.pi / 3)] + 2 * np.pi / 3
    new_thetas[np.logical_and(2 * np.pi / 3 <= thetas, thetas <= 4 * np.pi / 3)] = thetas[np.logical_and(2 * np.pi / 3 <= thetas, thetas <= 4 * np.pi / 3)]
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


thetas = np.linspace(-2 * np.pi / 3, 4 * np.pi / 3, 300)

nus = np.logspace(np.log10(1.0), np.log10(50.0), 36, base=10.0)
# Plot
n_rows = 6
n_cols = math.ceil(len(nus) / n_rows)
fig, ax = plt.subplots(n_rows, n_cols)
for i, nu in enumerate(nus):
    row_idx = i // n_cols
    col_idx = i % n_cols
    rhos = spherical_triangle(thetas, n=nu)
    x, y = pol2cart(rhos, thetas)
    # Plot and fill
    ax[row_idx, col_idx].fill(x, y, color='k')
    ax[row_idx, col_idx].set_aspect('equal')
    ax[row_idx, col_idx].set_axis_off()

plt.show()
