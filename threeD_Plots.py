import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Load the data
file_path = 'raw_3D.txt'

# Read the data while skipping the header rows and whitespace
data = pd.read_csv(file_path, delim_whitespace=True, skiprows=[1])

data_sorted = data.sort_values(by="Alpha")
# Extract relevant columns
alpha = data_sorted["Alpha"]  # Alpha (degrees)
fy = data_sorted["Fy"]        # Fy (N)



# Plot Fy vs Alpha
plt.figure(figsize=(8, 6))
plt.plot(alpha, fy, marker='s', linestyle='-', color='g', label='Experiment')
x_major_locator = MultipleLocator(5)  # Major ticks every 5 units on x-axis
y_major_locator = MultipleLocator(5)  # Major ticks every 5 units on y-axis

ax = plt.gca()  # Get current axes
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# Enable the grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.xlabel("Alpha [degrees]")
plt.ylabel("L [N]")
#plt.title("Fy vs Alpha")
plt.legend()
#plt.grid(True)
plt.show()