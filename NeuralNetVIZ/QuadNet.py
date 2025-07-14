import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Generate dataset: y = x^2
x = torch.linspace(-2, 2, 100).unsqueeze(1)
y = x**2

# Loss function
loss_fn = nn.MSELoss()


# Function to train model
def train_model(hidden_neurons, activation_fn, epochs=1000):
    model = nn.Sequential(
        nn.Linear(1, hidden_neurons), activation_fn(), nn.Linear(hidden_neurons, 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


# Try different neuron counts
neuron_counts = [1, 2, 3, 4, 5]
plt.figure(figsize=(12, 8))
for i, n in enumerate(neuron_counts, 1):
    model = train_model(n, nn.ReLU)
    with torch.no_grad():
        y_pred = model(x)

    plt.subplot(2, 3, i)
    plt.plot(x.numpy(), y.numpy(), label="True $x^2$", color="black")
    plt.plot(x.numpy(), y_pred.numpy(), label=f"Prediction (neurons={n})")
    plt.title(f"{n} Hidden Neurons")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.suptitle("Effect of Hidden Neuron Count on Learning $y = x^2$", fontsize=16, y=1.02)
plt.show()
