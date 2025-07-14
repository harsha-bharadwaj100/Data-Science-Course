import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Data: y = 2x + 1
x = np.linspace(0, 5, 100)
y_true = 2 * x + 1

# Training data (a few points)
x_train = np.array([0, 1, 2, 3, 4, 5])
y_train = 2 * x_train + 1

# Parameters
w = np.random.randn()
b = np.random.randn()
lr = 0.01

# Store weights over time for animation
history = []

# Train and record weights
for epoch in range(200):
    y_pred = w * x_train + b
    loss = ((y_pred - y_train) ** 2).mean()

    # Gradients
    dw = 2 * ((y_pred - y_train) * x_train).mean()
    db = 2 * (y_pred - y_train).mean()

    # Update
    w -= lr * dw
    b -= lr * db

    # Store line info
    history.append((w, b))

# 📽️ Animation
fig, ax = plt.subplots()
(line_true,) = ax.plot(x, y_true, "g--", label="True Function: 2x+1")
(line_pred,) = ax.plot([], [], "r-", label="Prediction")
scatter = ax.scatter(x_train, y_train, color="blue", label="Training Data")

ax.set_xlim(0, 5)
ax.set_ylim(0, 12)
ax.set_title("Neural Network Learning Visualization")
ax.legend()


def animate(i):
    w, b = history[i]
    y_pred = w * x + b
    line_pred.set_data(x, y_pred)
    ax.set_title(f"Epoch {i}, w={w:.2f}, b={b:.2f}")
    return (line_pred,)


ani = FuncAnimation(fig, animate, frames=len(history), interval=50, blit=True)

plt.show()

# Save the animation
ani.save("animation.mp4", writer="ffmpeg", fps=30)

print("Animation saved as 'animation.mp4'")
