import pandas as pd

# xlsl file with tap positions
file_path = 'SLT practical coordinates.xlsx'

#loading the file
excel_data = pd.ExcelFile(file_path) # Load the file
sheet_name = excel_data.sheet_names[0]
sheet_data = pd.read_excel(file_path, sheet_name=sheet_name) # Read the first sheet

#array containing the tap positions
rake_pos_taps_total_p = []
rake_pos_taps_static_p = []
# Get values from the useful columns
c=0
for col in sheet_data.columns:
    c+=1
    column_values = sheet_data[col].dropna().values # dropna in case NaN values are encountered
    match c:
        case 6: rake_pos_taps_total_p = column_values / 1000 #convert to m to work in SI
        case 9: rake_pos_taps_static_p = column_values / 1000


