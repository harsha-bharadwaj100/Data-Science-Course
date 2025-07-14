import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

# Activation functions
activations = {
    "ReLU": nn.ReLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
    "Identity": nn.Identity,
}

# Data
x = torch.linspace(-2, 2, 100).unsqueeze(1)
y = x**2
loss_fn = nn.MSELoss()


# Build model factory
def build_model(hidden_neurons, hidden_layers, act_fn_name):
    layers = []
    in_dim = 1
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_neurons))
        layers.append(activations[act_fn_name]())
        in_dim = hidden_neurons
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


# Train and return prediction + loss log
def train_model(model, lr, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_log = []
    for _ in range(epochs):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss_log.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model(x).detach(), loss_log


# Initial values
hidden_neurons = 4
hidden_layers = 1
act_fn_name = "ReLU"
lr = 0.01
epochs = 1000

# === PLOT SETUP ===
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(left=0.25, bottom=0.35)

# Initial training
model = build_model(hidden_neurons, hidden_layers, act_fn_name)
y_pred, loss_log = train_model(model, lr, epochs)

# Prediction plot
(true_line,) = ax1.plot(x, y, "k-", label="True y = x²")
(pred_line,) = ax1.plot(x, y_pred, "r-", label="NN Prediction")
ax1.set_title("Prediction vs True")
ax1.legend()
ax1.grid(True)

# Loss plot
(loss_line,) = ax2.plot(loss_log, "b-")
ax2.set_title("Training Loss")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("MSE Loss")
ax2.grid(True)

# === SLIDERS ===
# Axes for sliders
ax_lr = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_epochs = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_neurons = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_layers = plt.axes([0.25, 0.10, 0.65, 0.03])

# Sliders
s_lr = Slider(ax_lr, "LR", 0.001, 0.1, valinit=lr)
s_epochs = Slider(ax_epochs, "Epochs", 100, 3000, valinit=epochs, valstep=100)
s_neurons = Slider(ax_neurons, "Neurons", 1, 10, valinit=hidden_neurons, valstep=1)
s_layers = Slider(ax_layers, "Layers", 1, 5, valinit=hidden_layers, valstep=1)

# === ACTIVATION BUTTONS ===
ax_act = plt.axes([0.025, 0.5, 0.15, 0.15])
act_button = Button(ax_act, f"Act: {act_fn_name}")


# === UPDATE FUNCTION ===
def update(val=None):
    global act_fn_name
    model = build_model(int(s_neurons.val), int(s_layers.val), act_fn_name)
    y_pred, loss_log = train_model(model, s_lr.val, int(s_epochs.val))

    pred_line.set_ydata(y_pred)
    loss_line.set_ydata(loss_log)
    loss_line.set_xdata(np.arange(len(loss_log)))

    ax2.relim()
    ax2.autoscale_view()
    fig.canvas.draw_idle()


# === BUTTON HANDLER ===
def change_activation(event):
    global act_fn_name
    keys = list(activations.keys())
    idx = (keys.index(act_fn_name) + 1) % len(keys)
    act_fn_name = keys[idx]
    act_button.label.set_text(f"Act: {act_fn_name}")
    update()


# === Event bindings ===
s_lr.on_changed(update)
s_epochs.on_changed(update)
s_neurons.on_changed(update)
s_layers.on_changed(update)
act_button.on_clicked(change_activation)

# Show the interactive plot
plt.show(block=True)
