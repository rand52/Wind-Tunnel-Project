import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# Example data
x = np.linspace(-10, 50, 100)
y = np.sin(x)

# Create the plot
plt.plot(x, y)

# Set up the grid with equal spacing of 5 units on both axes
major_locator = MultipleLocator(5)  # Spacing of 5 units for major ticks

ax = plt.gca()  # Get current axes
ax.xaxis.set_major_locator(major_locator)  # Apply to x-axis
ax.yaxis.set_major_locator(MultipleLocator(0.5))  # Apply to y-axis

# Enable the grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')

plt.show()