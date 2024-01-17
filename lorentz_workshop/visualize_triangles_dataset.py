import math

import numpy as np
import matplotlib.pyplot as plt

from create_triangle_dataset import spherical_triangle, pol2cart, rotate, plot_triangle


thetas = np.linspace(-2 * np.pi / 3, 4 * np.pi / 3, 300)
canon_theta = np.deg2rad(-30)

colormap = plt.cm.gray
# nus = np.logspace(np.log10(1e-6), np.log10(1.0), 36, base=10.0)
# nus = np.logspace(-1.5, 1.2, 36)
nus = np.linspace(1e-3, 1.0, 36)
# Plot
n_rows = 6
n_cols = math.ceil(len(nus) / n_rows)
fig, ax = plt.subplots(n_rows, n_cols, facecolor="black")
for i, nu in enumerate(nus):
    row_idx = i // n_cols
    col_idx = i % n_cols

    rhos = spherical_triangle(thetas, nu=nu)
    x, y = pol2cart(rhos, thetas)
    # Rotate
    rand_theta = np.random.uniform(0, 2 * np.pi)
    x, y = rotate(x, y, canon_theta + rand_theta)
    # Random color, don't sample completely black
    rand_color = np.random.randint(20, 255)
    # Fill
    plot_triangle(ax[row_idx, col_idx], x, y, color=rand_color)

plt.show()
