import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------- 1. Data ----------
np.random.seed(42)
X = np.random.randint(0, 100, (1000, 2))
y = X[:, 0] + X[:, 1]

X = X / 100.0  # min–max scale inputs
y = y / 200.0  # min–max scale outputs

# ---------- 2. Model ----------
w = np.random.randn(2)
b = np.random.randn()
lr = 0.1

history = []
for epoch in range(1000):
    pred = X @ w + b
    loss = ((pred - y) ** 2).mean()

    gw = 2 * (X.T @ (pred - y)) / len(X)
    gb = 2 * (pred - y).mean()
    w -= lr * gw
    b -= lr * gb

    if epoch % 10 == 0:
        history.append((w.copy(), b))

# ---------- 3. Surfaces ----------
a_vals = np.linspace(0, 100, 50)
b_vals = np.linspace(0, 100, 50)
A, B = np.meshgrid(a_vals, b_vals)
true_Z = A + B
grid_inputs = np.c_[A.ravel(), B.ravel()] / 100.0

# ---------- 4. Figure ----------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("Input A")
ax.set_ylabel("Input B")
ax.set_zlabel("Sum (A + B)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 200)

# Static true plane
ax.plot_surface(A, B, true_Z, color="red", alpha=0.25)

# First (random) prediction surface
w0, b0 = history[0]
pred_Z0 = (grid_inputs @ w0 + b0) * 200.0
pred_Z0 = pred_Z0.reshape(A.shape)
pred_surf = ax.plot_surface(A, B, pred_Z0, cmap="viridis", alpha=0.8)


def update(frame):
    """Update the predicted surface each frame."""
    global pred_surf
    # Drop old surface
    pred_surf.remove()

    # New surface
    w, b = history[frame]
    pred_Z = (grid_inputs @ w + b) * 200.0
    pred_Z = pred_Z.reshape(A.shape)
    pred_surf = ax.plot_surface(A, B, pred_Z, cmap="viridis", alpha=0.8)

    ax.set_title(f"Epoch {frame*10} — w≈[{w[0]:.2f}, {w[1]:.2f}], b≈{b:.2f}")
    return (pred_surf,)


anim = FuncAnimation(fig, update, frames=len(history), interval=50, blit=False)

# ---------- 5. Save (optional) ----------
# Uncomment if you want a file (requires ffmpeg installed & on PATH)
anim.save("add_net_learning.mp4", writer="ffmpeg", fps=30, dpi=150)

plt.tight_layout()
plt.show()
