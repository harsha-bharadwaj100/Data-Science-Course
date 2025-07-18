import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Generate noisy circle data
def generate_noisy_circle(radius=1.0, num_points=300, noise=0.05):
    angles = 2 * np.pi * np.random.rand(num_points)
    x = radius * np.cos(angles) + np.random.normal(0, noise, num_points)
    y = radius * np.sin(angles) + np.random.normal(0, noise, num_points)
    X = np.stack([x, y], axis=1)
    Y = np.zeros((num_points, 1))
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
        x,
        y,
    )


# Simple MLP
class ImplicitCircleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.SiLU(),  # Sigmoid Linear Unit
            nn.Linear(8, 8),
            nn.SiLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


# Data
X_tensor, Y_tensor, x_raw, y_raw = generate_noisy_circle()
model = ImplicitCircleNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Grid for contour plotting
grid_res = 80
xv = np.linspace(-1.5, 1.5, grid_res)
yv = np.linspace(-1.5, 1.5, grid_res)
xg, yg = np.meshgrid(xv, yv)
grid = np.stack([xg.ravel(), yg.ravel()], axis=1)
grid_tensor = torch.tensor(grid, dtype=torch.float32)

# Setup animation
fig, ax = plt.subplots(figsize=(5, 5))


def init():
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_title("Learning a Noisy Circle")
    ax.grid(True)


def update(frame):
    for _ in range(5):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = loss_fn(y_pred, Y_tensor)
        loss.backward()
        optimizer.step()

    ax.clear()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_title(f"Epoch {frame*5}, Loss: {loss.item():.5f}")
    ax.grid(True)

    with torch.no_grad():
        z = model(grid_tensor).numpy().reshape(grid_res, grid_res)
    ax.contour(xg, yg, z, levels=[0], colors="red")
    ax.scatter(x_raw, y_raw, s=2, color="blue")


ani = FuncAnimation(fig, update, frames=240, init_func=init)

# Save the animation
ani.save("CircleFit.mp4", writer="ffmpeg", fps=30)
print("Animation saved as 'CircleFit.mp4'")
plt.show()
