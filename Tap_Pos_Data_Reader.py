import pandas as pd
import numpy as np

# xlsx file with tap positions
file_path = 'SLT practical coordinates.xlsx'

#loading the file
excel_data = pd.ExcelFile(file_path)  # Load the file
sheet_name = excel_data.sheet_names[0]
sheet_data = pd.read_excel(file_path, sheet_name=sheet_name)  # Read the first sheet

#start position of the top nd bottom airfoil taps in the column
af_top_coords_start = 1
af_bottom_coords_start = 26

#array containing the tap positions
rake_pos_taps_total_p: np.ndarray
rake_pos_taps_static_p: np.ndarray
airfoil_pos_top_taps: np.ndarray
airfoil_pos_bottom_taps: np.ndarray
# Get values from the useful columns
c = 0
for col in sheet_data.columns:
    c += 1
    column_values = sheet_data[col].dropna().values  # dropna in case NaN values are encountered
    match c:
        case 2:
            airfoil_pos_top_taps = column_values[af_top_coords_start:af_bottom_coords_start:]  # in % of chord
            airfoil_pos_bottom_taps = column_values[af_bottom_coords_start::]
        case 6:
            rake_pos_taps_total_p = column_values / 1000  # /1000 to convert to m and work in SI
        case 9:
            rake_pos_taps_static_p = column_values / 1000
