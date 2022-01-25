## Geometric shapes dataset <img width="20" alt="SimpleGeometric Logo" src="https://user-images.githubusercontent.com/3244249/150962823-39590ae2-4b0c-4536-9159-60ada2207d11.png"> generation

The geometric shapes dataset is a custom dataset created specifically for DIANNA. The [generate_geometric_shapes.ipynb](generate_geometric_shapes.ipynb) notebook handles the generation of this dataset. It creates a user-specified amount of circles and triangles, stored as 64x64 pixel grayscale images in an `shapes.npz` file. The circles have label 0, while the triangles have label 1.
