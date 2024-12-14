import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Load the data
file_path = 'raw_3D.txt'

# Read the data while skipping the header rows and whitespace
data = pd.read_csv(file_path, delim_whitespace=True, skiprows=[1])

#data_sorted = data.sort_values(by="Alpha")
data_sorted = data

# Extract relevant columns
alpha = data_sorted["Alpha"]  # Alpha (degrees)
fy = data_sorted["Fy"]        # Fy (N)
fx = data_sorted["Fx"]        # Fx (N)
CL = fy / (0.5 * 1.19 * 20**2 * 0.064)
CD = fx / (0.5 * 0.19 * 20**2 * 0.064)
CDi = CL**2 / (math.pi * 5) * (1+0.0619)

Slope3D = (CL[14]-CL[0]) * (180 / math.pi) / (14)

print(Slope3D)

# Plot Fy vs Alpha
plt.figure(figsize=(8, 6))
plt.plot(alpha, CL, marker='s', markersize=3, linestyle='-', color='g', label='Experiment')
x_major_locator = MultipleLocator(5)  # Major ticks every 5 units on x-axis
y_major_locator = MultipleLocator(0.1)  # Major ticks every 5 units on y-axis

ax = plt.gca()  # Get current axes
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# Enable the grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.xlabel("\u03B1 [degrees]")
plt.ylabel("$C_L$")
#plt.title("Fy vs Alpha")
plt.legend(loc="upper left", bbox_to_anchor=(0.1, 0.8))
#plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(CD, CL, marker='s', markersize=3, linestyle='-', color='g', label='Experiment')
plt.plot(CDi, CL, marker='s', markersize=3, linestyle='-', color='g', label='Experiment')
x_major_locator = MultipleLocator(0.2)  # Major ticks every 5 units on x-axis
y_major_locator = MultipleLocator(0.2)  # Major ticks every 5 units on y-axis

ax = plt.gca()  # Get current axes
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# Enable the grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.xlabel("$C_D$")
plt.ylabel("$C_L$")
#plt.title("Fy vs Alpha")
plt.legend(loc="upper left", bbox_to_anchor=(0.6, 0.5))
#plt.grid(True)
plt.show()