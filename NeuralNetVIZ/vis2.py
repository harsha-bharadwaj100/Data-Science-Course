import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Generate noisy circular data (x^2 + y^2 ≈ 1)
np.random.seed(0)
n_points = 500
theta = np.random.uniform(0, 2 * np.pi, n_points)
r = 1 + np.random.normal(0, 0.05, n_points)
x_data = r * np.cos(theta)
y_data = r * np.sin(theta)
z_data = np.zeros(n_points)  # We want f(x, y) ≈ 0 on the circle

# Convert to tensors
X = torch.tensor(np.vstack([x_data, y_data]).T, dtype=torch.float32)
Y = torch.tensor(z_data[:, None], dtype=torch.float32)


# 2. Define simple MLP
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 3. Set up plot
fig, ax = plt.subplots()
grid_x, grid_y = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
grid_tensor = torch.tensor(grid, dtype=torch.float32)
contour = None
scatter = ax.scatter(x_data, y_data, c="red", s=10, label="Noisy Circle Points")


# 4. Animation function
def update(frame):
    global contour
    for _ in range(5):  # Train a few steps per frame
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(grid_tensor).reshape(100, 100).numpy()

    if contour:
        contour.remove()
    contour = ax.contour(grid_x, grid_y, preds, levels=[0], colors="blue")
    ax.set_title(f"Epoch {frame*5}, Loss: {loss.item():.4f}")
    return contour


# 5. Run animation
anim = FuncAnimation(fig, update, frames=240, interval=100)
anim.save("circle_fit.mp4", writer="ffmpeg", fps=12, dpi=150)
plt.legend()
plt.show()

# 6. Save animation (requires ffmpeg installed & on PATH)

print("Animation saved as 'circle_fit.mp4'")
