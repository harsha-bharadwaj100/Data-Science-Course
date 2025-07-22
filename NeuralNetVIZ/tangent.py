# import numpy as np
# import matplotlib.pyplot as plt

# # Parametric curve that satisfies the constant tangent length condition
# t = np.linspace(-10, 10, 1000)
# l = 1

# x = t + l / np.sqrt(1 + t**2)
# y = l * t / np.sqrt(1 + t**2)

# plt.figure(figsize=(10, 5))
# plt.plot(x, y, label="Curve with constant tangent length to x-axis (l=1)")
# plt.axhline(0, color='gray', linestyle='--')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.axis('equal')
# plt.grid(True)
# plt.title("Curve Where Every Tangent Hits X-axis at Fixed Distance")
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Define the constant length 'a' (which is 'k' from the derivation)
a = 1.0

# Generate a range of 't' values
# Avoid t=0 and t=pi due to log(tan(t/2)) becoming undefined or infinite
t = np.linspace(0.01, np.pi - 0.01, 500)

# Calculate x and y using the parametric equations of the tractrix
x = a * (np.log(np.tan(t / 2)) + np.cos(t))
y = a * np.sin(t)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=f"Tractrix (Constant Tangent Length = {a})")

# Add labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Plot of a Tractrix")
plt.grid(True)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.legend()
plt.ylim(bottom=0)  # Ensure y-axis starts from 0 as y = a sin(t) will be positive

# Save the plot
plt.savefig("tractrix_plot.png")

print("Tractrix plot saved as tractrix_plot.png")
