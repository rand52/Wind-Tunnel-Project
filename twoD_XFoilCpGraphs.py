import matplotlib.pyplot as plt
import pandas as pd

# Specify the file path
file_path = 'XFoil_Data.xlsx'

# Read the Excel sheet
data = pd.read_excel(file_path)


# Function to split data into upper and lower surfaces
def split_upper_lower(data):
    """
    Splits x/c and Cp data into upper and lower surfaces based on symmetry.

    Args:
        data (DataFrame): Data containing x/c and Cp values.

    Returns:
        x_upper, cp_upper, x_lower, cp_lower: Separate arrays for upper and lower surfaces.
    """
    midpoint = len(data) // 2  # Approximation of leading edge index
    x_upper = data.iloc[:midpoint, 1]  # First half x-coordinates
    cp_upper = data.iloc[:midpoint, 2]  # First half Cp values
    x_lower = data.iloc[midpoint:, 1]  # Second half x-coordinates
    cp_lower = data.iloc[midpoint:, 2]  # Second half Cp values
    return x_upper, cp_upper, x_lower, cp_lower


# Extract unique angles of attack
angles = data['alpha(deg)'].dropna().unique()

# Use the original plotting style you provided
plt.rcParams['font.family'] = 'Palatino Linotype'

# Iterate through each angle of attack and plot the data
plt.figure(figsize=(10, 6))

for alpha in angles:
    # Filter data for the current angle of attack
    angle_data = data[data['alpha(deg)'] == alpha].reset_index(drop=True)

    # Split into upper and lower surfaces
    x_upper, cp_upper, x_lower, cp_lower = split_upper_lower(angle_data)

    # Plot upper surface
    plt.plot(
        x_upper, cp_upper,
        label=f'Upper surface (α={alpha}°)', marker='o', color='black', linestyle='-',
        linewidth=1, markersize=6, markeredgecolor='black', markerfacecolor='cyan'
    )

    # Plot lower surface
    plt.plot(
        x_lower, cp_lower,
        label=f'Lower surface (α={alpha}°)', marker='^', color='black', linestyle='-',
        linewidth=1, markersize=6, markeredgecolor='black', markerfacecolor=(1.0, 0.078, 0.576)
    )

# Configure plot settings
plt.gca().invert_yaxis()  # Invert Cp axis
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.xlabel("x/c (%)", fontsize=16)
plt.ylabel(r"$c_p$  (-)", fontsize=16)

plt.tight_layout()
plt.show()
