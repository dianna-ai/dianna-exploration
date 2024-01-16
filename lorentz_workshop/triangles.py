import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np

sqrt_2 = math.sqrt(2)
sqrt_3 = math.sqrt(3)
X = np.array([[0.0, 0.0], [1., 0.0], [0.5, 0.5 * sqrt_3]])
center = X.mean(0, keepdims=True)
X -= center
center = np.array([0.0, 0.0])

# triangle = mpl.patches.Polygon(X)

arc1 = mpl.patches.Arc(X[0], 2, 2, theta1=0, theta2=60)
arc2 = mpl.patches.Arc(X[1], 2, 2, theta1=120, theta2=180)
arc3 = mpl.patches.Arc(X[2], 2, 2, theta1=240, theta2=300)

p = PatchCollection([arc1, arc2, arc3], alpha=0.4)
p.set_array([0.0, 0.0, 0.0])
fig, ax = plt.subplots()
ax.add_collection(p)
ax.set_aspect('equal')
ax.scatter(center[0], center[1], color='r')
ax.set_axis_off()

plt.show()
# plt.savefig('triangles.png', bbox_inches='tight')
