"""
Inspired by https://math.stackexchange.com/questions/786335/is-there-a-function-that-draws-a-triangle-with-rounded-edges
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def ang(thetas):
    new_thetas = np.zeros_like(thetas)
    new_thetas[np.logical_and(0 <= thetas, thetas < 2 * np.pi / 3)] = (
        thetas[np.logical_and(0 <= thetas, thetas < 2 * np.pi / 3)] + 2 * np.pi / 3
    )
    new_thetas[np.logical_and(2 * np.pi / 3 <= thetas, thetas <= 4 * np.pi / 3)] = (
        thetas[np.logical_and(2 * np.pi / 3 <= thetas, thetas <= 4 * np.pi / 3)]
    )
    new_thetas[np.logical_and(-4 * np.pi / 3 < thetas, thetas < 0)] = (
        thetas[np.logical_and(-4 * np.pi / 3 < thetas, thetas < 0)] - 2 * np.pi / 3
    )
    return new_thetas


def polar_triangle(thetas):
    return -1.0 / (2.0 * np.cos(ang(thetas)))


def spherical_triangle(thetas, nu=1):
    """
    Create a spherical triangle with rounded edges

    As nu decreases, the triangle becomes more and more rounded
    When nu -> 0, the triangle becomes a circle
    When nu = 1, the triangle is a regular triangle
    When nu > 1, the triangle becomes hyperbolic
    """
    return polar_triangle(thetas) ** nu


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def rotate(x, y, theta):
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    return x_rot, y_rot


def plot_triangle(ax, x, y, color, color_map=plt.cm.gray):
    ax.fill(x, y, color=color_map(color))
    ax.set_aspect("equal")
    ax.set_axis_off()


def dataset_instance(thetas, instance, variation):
    rhos = spherical_triangle(
        thetas,
        nu=(
            variation if instance == "roundedness" else default_instances["roundedness"]
        ),
    )
    x, y = pol2cart(rhos, thetas)
    x, y = rotate(
        x, y, variation if instance == "rotation" else default_instances["rotation"]
    )
    color = variation if instance == "color" else default_instances["color"]
    return x, y, color


thetas = np.linspace(-2 * np.pi / 3, 4 * np.pi / 3, 300)
canon_theta = np.deg2rad(-30)
COLORMAP = plt.cm.gray

# Create `num_variations` versions of the dataset, each one with a different
# axis of variation, including color, rotation, and roundedness.
num_samples = 100
use_linspace = True
if use_linspace:
    variations = {
        "color": np.linspace(20, 255, num_samples).astype(int),
        "rotation": canon_theta + np.linspace(0, 2 * np.pi, num_samples),
        "roundedness": np.linspace(1e-3, 1.0, num_samples),
    }
else:
    variations = {
        "color": np.random.randint(20, 255, size=(num_samples,)),
        "rotation": canon_theta + np.random.uniform(0, 2 * np.pi, size=(num_samples,)),
        "roundedness": np.random.uniform(1e-3, 1, size=(num_samples,)),
    }
default_instances = {
    "color": 255,
    "rotation": canon_theta,
    "roundedness": 1.0,
}
default_instance_arrays = {
    "color": np.full_like(variations["color"], 1.0),
    "rotation": np.full_like(variations["rotation"], 0.0),
    "roundedness": np.full_like(variations["roundedness"], 1.0),
}

# Remove variations you aren't interested in
variations_to_use = [
    "color",
    "rotation",
    "roundedness",
]
dpi = 300
width = 64
height = 64
datasets = {}
for key, values in variations.items():
    imgs = []
    for value in values:
        x, y, color = dataset_instance(thetas, key, value)
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
    # Save all variation axes, 2 of them as placeholder each time
    roundedness = (
        values if key == "roundedness" else default_instance_arrays["roundedness"]
    )
    color = values / 255.0 if key == "color" else default_instance_arrays["color"]
    rotation = values if key == "rotation" else default_instance_arrays["rotation"]
    datasets[key] = {
        "images": imgs.astype(np.float32),
        "roundedness": roundedness.astype(np.float32),
        "color": color.astype(np.float32),
        "rotation": rotation.astype(np.float32),
        "labels": roundedness > 0.1,
    }

# Save each key as a separate file
for key, value in datasets.items():
    np.savez_compressed(f"dataset_{key}.npz", **value)
