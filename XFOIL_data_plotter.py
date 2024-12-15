import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


class twoD_DP_experimental:
    reynolds_num = 2.1 * 10 ** 5  # reynolds number constant for all experiments

    def __init__(self):
        """Constructor for the 2D experimental datapoint class. Takes experimental data for a certain AOA"""
        self.aoa: float = None
        self.top_coordinates: np.ndarray = None
        self.top_cps: np.ndarray = None
        self.bottom_coordinates: np.ndarray = None
        self.bottom_cps: np.ndarray = None

    def plot_Cp(self, save: bool = False):
        """Save = False doesn't save plots
        save = True saves plots to directory"""
        # plot the Cps
        plt.figure(figsize=(7.8, 6))
        # Plot top side pressures
        plt.plot(
            self.top_coordinates,
            self.top_cps,
            color="red",
            marker='.',
            label='Upper Surface (Red)',
            linewidth=1,
            markeredgecolor='black',
            markersize=10
        )
        # Plot bottom side pressures
        plt.plot(
            self.bottom_coordinates,
            self.bottom_cps,
            color="green",
            marker='s',
            label='Lower Surface (Green)',
            linewidth=1,
            markeredgecolor='black',
            markersize=5,
            linestyle='-'  # Ensure lines connect the points
        )
        # Display the minimum Cp in the top-left corner below legend
        plt.text(
            0.98, 0.85,
            fr"$\alpha$ = {self.aoa:.2f}°",  # format in scientific notation
            transform=plt.gca().transAxes,
            fontsize=10,
            horizontalalignment='right'
        )
        plt.text(
            0.98, 0.80,
            f"$C_{{p,\\text{{min}}}}$ : {min(np.min(self.top_cps), np.min(self.bottom_cps)):.2f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            horizontalalignment='right'
        )
        # Display the maximum Cp in the top-left corner below legend
        plt.text(
            0.98, 0.75,
            f"$C_{{p,\\text{{max}}}}$: {max(np.max(self.top_cps), np.max(self.bottom_cps)):.2f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            horizontalalignment='right'
        )
        # Display the reynolds_number in the top-left corner below legend
        plt.text(
            0.98, 0.70,
            f"Re = {self.reynolds_num:.2e}",  # format in scientific notation
            transform=plt.gca().transAxes,
            fontsize=10,
            horizontalalignment='right'
        )
        # Invert the y-axis as it's a Cp plot
        plt.gca().invert_yaxis()
        # Labeling the plot
        plt.xlabel("Position along chord (x/c) [%]", fontsize=12)
        plt.ylabel(r"$C_p$ [-]", fontsize=12)
        plt.legend()
        # Make a grid in the background for better readability
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Reference line for Cp = 0
        plt.grid(True, linestyle='--', alpha=0.6)
        # Save the plot to the specified directory
        if save:
            os.makedirs("Plots", exist_ok=True)  # Ensure the directory exists
            file_path = os.path.join(r"Plots", f"2D_Cp_Experimental_AOA_{self.aoa}.png")
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1)
        # Display the plot, after saving it
        plt.show()
        plt.close()  # Close the figure to free memory


# xlsx file with tap positions
file_path = 'XFoil_Data.xlsx'

# loading the file
excel_data = pd.ExcelFile(file_path)  # Load the file
sheet_name = excel_data.sheet_names[0]
sheet_data = pd.read_excel(file_path, sheet_name=sheet_name)  # Read the first sheet

# start position of the top and bottom airfoil taps in the column
af_top_data_start = 1
af_bottom_data_start = 81

# list containing the datapoints
datapoints = []

# Get value set from 3 columns
c = 0  # keep track of column count
datPt: twoD_DP_experimental = None  # current datapoint
for col in sheet_data.columns:
    column_values = sheet_data[col].dropna().values  # dropna in case NaN values are encountered
    if c % 3 == 0:
        datPt = twoD_DP_experimental()  # new datapoint every 3 columns
    match c % 3:
        case 0:
            datPt.aoa = column_values[af_top_data_start]
        case 1:
            datPt.top_coordinates = column_values[af_top_data_start:af_bottom_data_start:] * 100  # in % of chord
            datPt.bottom_coordinates = column_values[af_bottom_data_start::] * 100
        case 2:
            datPt.top_cps = column_values[af_top_data_start:af_bottom_data_start:]  # in % of chord
            datPt.bottom_cps = column_values[af_bottom_data_start::]
    if c % 3 == 2:
        datapoints.append(datPt)  # after a datapoint is read append it
    c += 1

for i in datapoints:
    i.plot_Cp()