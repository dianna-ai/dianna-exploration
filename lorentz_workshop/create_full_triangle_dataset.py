"""
Inspired by https://math.stackexchange.com/questions/786335/is-there-a-function-that-draws-a-triangle-with-rounded-edges
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from create_triangle_dataset import ang, polar_triangle, spherical_triangle, pol2cart, rotate, plot_triangle


thetas = np.linspace(-2 * np.pi / 3, 4 * np.pi / 3, 300)
canon_theta = np.deg2rad(-30)
COLORMAP = plt.cm.gray

dpi = 300
width = 64
height = 64
datasets = {}

full_dataset_size = 5000
full_dataset = {}
variations = {
    "color": np.random.randint(20, 255, size=(full_dataset_size,)),
    "rotation": canon_theta + np.random.uniform(0, 2 * np.pi, size=(full_dataset_size,)),
    "roundedness": np.random.uniform(1e-3, 1, size=(full_dataset_size,)),
}
imgs = []
for i in range(full_dataset_size):
    rhos = spherical_triangle(thetas, nu=variations["roundedness"][i])
    x, y = pol2cart(rhos, thetas)
    x, y = rotate(x, y, variations["rotation"][i])
    color = variations["color"][i]

    fig, ax = plt.subplots(facecolor="black")
    plot_triangle(ax, x, y, color)

    plt.savefig("tmp.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    img = (
        np.array(Image.open("tmp.png").convert("L").resize((width, height))) / 255.0
    )

    imgs.append(img)
    # Close figure
    plt.close(fig)
imgs = np.stack(imgs, axis=0)

train_test_split = 0.9
num_train = int(full_dataset_size * train_test_split)
num_test = full_dataset_size - num_train

full_dataset = {
    "X_train": imgs.astype(np.float32)[:num_train],
    "y_train": (variations['roundedness'] > 0.1).astype(np.int32)[:num_train],
    "roundedness_train": variations['roundedness'].astype(np.float32)[:num_train],
    "color_train": (variations['color'] / 255.0).astype(np.float32)[:num_train],
    "rotation_train": variations['rotation'].astype(np.float32)[:num_train],
    # Test
    "X_test": imgs.astype(np.float32)[num_train:],
    "y_test": (variations['roundedness'] > 0.1).astype(np.int32)[num_train:],
    "roundedness_test": variations['roundedness'].astype(np.float32)[num_train:],
    "color_test": (variations['color'] / 255.0).astype(np.float32)[num_train:],
    "rotation_test": variations['rotation'].astype(np.float32)[num_train:],
}

# Save full dataset
np.savez_compressed("full_dataset.npz", **full_dataset)
