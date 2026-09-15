import numpy as np
import matplotlib.pyplot as plt


# For reproducible random values
np.random.seed(1)

# Number of random points
n_samples = 300

# Radius of the circle
radius = 2.0

# Generate random x1 and x2 values
X = np.random.uniform(-4, 4, size=(n_samples, 2))

# Extract the two features
x1 = X[:, 0]
x2 = X[:, 1]

# Calculate squared distance of each point from the origin
distance_squared = x1**2 + x2**2

# Assign class labels
# Inside the circle: class 0
# Outside the circle: class 1
y = np.where(distance_squared <= radius**2, 0, 1)


# Plot the points
plt.figure(figsize=(8, 8))

plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    color='blue',
    marker='o',
    label='Class 0: Inside circle'
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    color='red',
    marker='x',
    label='Class 1: Outside circle'
)


# Draw the circular decision boundary
theta = np.linspace(0, 2 * np.pi, 500)

circle_x = radius * np.cos(theta)
circle_y = radius * np.sin(theta)

plt.plot(
    circle_x,
    circle_y,
    color='black',
    linewidth=2,
    label='Nonlinear decision boundary'
)


# Mark the origin
plt.scatter(0, 0, color='black', marker='+', s=100)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Nonlinearly Separable Data')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.grid(True)
plt.show()