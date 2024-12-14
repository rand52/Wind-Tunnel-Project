import matplotlib.pyplot as plt
import pandas as pd

# Specify the file path
file_path = 'XFoil_Data.xlsx'

# Read the Excel sheet
data = pd.read_excel(file_path)


# Function to split data into upper and lower surfaces
def split_upper_lower(x, cp):
    """
    Splits x/c and Cp data into upper and lower surfaces based on symmetry.

    Args:
        x (Series): x coordinates.
        cp (Series): Cp values.

    Returns:
        x_upper, cp_upper, x_lower, cp_lower: Separate arrays for upper and lower surfaces.
    """
    midpoint = len(x) // 2  # Approximation of leading edge index
    x_upper = x[:midpoint]  # First half x-coordinates
    cp_upper = cp[:midpoint]  # First half Cp values
    x_lower = x[midpoint:]  # Second half x-coordinates
    cp_lower = cp[midpoint:]  # Second half Cp values
    return x_upper, cp_upper, x_lower, cp_lower


# Iterate through the sets of 3 columns: angle, x, Cp
num_angles = len(data.columns) // 3  # Total number of angles (since each angle has 3 columns)

# Iterate through each angle
for i in range(num_angles):
    # Extract the relevant columns for the current angle (set of 3 columns)
    angle_column = data.columns[i * 3]  # e.g., alpha(deg)_1, alpha(deg)_2, etc.
    x_column = data.columns[i * 3 + 1]  # x coordinates
    cp_column = data.columns[i * 3 + 2]  # Cp values

    # Extract unique angles for this set of columns
    angles = data[angle_column].dropna().unique()

    # Create a new figure for each angle set
    plt.figure(figsize=(10, 6))

    for alpha in angles:
        # Filter data for the current angle
        angle_data = data[data[angle_column] == alpha].reset_index(drop=True)

        # Extract x and Cp values for the current angle
        x = angle_data[x_column].values
        cp = angle_data[cp_column].values

        # Split into upper and lower surfaces
        x_upper, cp_upper, x_lower, cp_lower = split_upper_lower(x, cp)

        # Plot upper surface for the current angle
        plt.plot(
            x_upper, cp_upper,
            label=f'Upper surface (α={alpha}°)', marker='o', color='black', linestyle='-',
            linewidth=1, markersize=6, markeredgecolor='black', markerfacecolor='cyan'
        )

        # Plot lower surface for the current angle
        plt.plot(
            x_lower, cp_lower,
            label=f'Lower surface (α={alpha}°)', marker='^', color='black', linestyle='-',
            linewidth=1, markersize=6, markeredgecolor='black', markerfacecolor=(1.0, 0.078, 0.576)
        )

    # Configure plot settings for the current figure
    plt.gca().invert_yaxis()  # Invert Cp axis
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.xlabel("x/c (%)", fontsize=16)
    plt.ylabel(r"$c_p$  (-)", fontsize=16)

    # Tight layout for better spacing
    plt.tight_layout()

    # Show the plot for the current set of angles
    plt.show()

