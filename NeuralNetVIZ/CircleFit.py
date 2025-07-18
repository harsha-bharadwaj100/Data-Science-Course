import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Circle parameters
radius = 1.0
num_points = 500

# Generate circle data: points on the circle (x² + y² = r²)
angles = 2 * np.pi * np.random.rand(num_points)
x = radius * np.cos(angles)
y = radius * np.sin(angles)
X = np.stack([x, y], axis=1)
Y = np.zeros((num_points, 1))  # All targets are 0, since x² + y² - r² = 0 on the circle

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)


# Simple implicit function neural network
class ImplicitCircleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 1)
        )

    def forward(self, input):
        return self.net(input)


model = ImplicitCircleNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(X_tensor)
    loss = loss_fn(y_pred, Y_tensor)
    loss.backward()
    optimizer.step()

# Visualize: evaluate on grid
grid_res = 100
xv = np.linspace(-1.5, 1.5, grid_res)
yv = np.linspace(-1.5, 1.5, grid_res)
xg, yg = np.meshgrid(xv, yv)
grid = np.stack([xg.ravel(), yg.ravel()], axis=1)
grid_tensor = torch.tensor(grid, dtype=torch.float32)

# Get predictions and reshape for contour plot
with torch.no_grad():
    pred_vals = model(grid_tensor).numpy().reshape(grid_res, grid_res)

# Plot the implicit contour
plt.figure(figsize=(6, 6))
plt.contour(xg, yg, pred_vals, levels=[0], colors="red")
plt.scatter(x, y, s=1, label="Training Points")
plt.title("Implicit Regression: Approximated Circle")
plt.gca().set_aspect("equal")
plt.legend()
plt.grid(True)
plt.show()
