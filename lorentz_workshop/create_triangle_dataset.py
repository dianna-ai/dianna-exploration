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


def rotate(x, y, theta):
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    return x_rot, y_rot


thetas = np.linspace(-2 * np.pi / 3, 4 * np.pi / 3, 300)

colormap = plt.cm.gray
nus = np.logspace(np.log10(1.0), np.log10(50.0), 36, base=10.0)
# Plot
n_rows = 6
n_cols = math.ceil(len(nus) / n_rows)
fig, ax = plt.subplots(n_rows, n_cols, facecolor='black')
for i, nu in enumerate(nus):
    row_idx = i // n_cols
    col_idx = i % n_cols
    rhos = spherical_triangle(thetas, n=nu)
    x, y = pol2cart(rhos, thetas)
    rand_theta = np.random.uniform(0, 2 * np.pi)
    x, y = rotate(x, y, rand_theta)
    # Plot and fill
    rand_color = np.random.randint(0, 255)
    ax[row_idx, col_idx].fill(x, y, color=colormap(rand_color))
    ax[row_idx, col_idx].set_aspect('equal')
    ax[row_idx, col_idx].set_axis_off()

plt.show()
