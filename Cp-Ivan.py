import numpy as np
import pandas as pd
import scipy as sc

import twoD_Datapoint as dp

# Load the Excel file
file_path = "SLT practical coordinates.xlsx"
df = pd.read_excel(file_path)

# Specify the column name
column_name = "total wake rake probe locations [mm]"  # Replace with your column name

# Check if the column exists
if column_name in df.columns:
    # Select only numeric values
    numeric_data = df[column_name].dropna()  # Remove NaN values
    numeric_data = numeric_data[pd.to_numeric(numeric_data, errors="coerce").notnull()]
    print(numeric_data)
else:
    print(f"Column '{column_name}' does not exist.")



# txt datafile with test results
datafile = "raw_2D.txt"
comment_indicator = "#"  # line indicator NOT read
header = 2  # header NOT read size

# Read the file into a Pandas DataFrame
df = pd.read_csv(
    datafile,
    sep=r"\s+", # delimiter = whitespaces
    comment=comment_indicator,  # line indicator NOT read
    skiprows=header,  # don't read header lines
    header=None  # use int headers for columns and rows
)

# list with the datapoints
datapoints = []

for row in df.values:
    datPt = dp.twoD_DP()  # create new datapoint
    # save data to the datapoint
    datPt.aoa = row[2]
    datPt.del_pb = row[3]
    datPt.p_atm = row[4] * 100  # it's given in kPa
    datPt.temp_C = row[5]
    datPt.rho_st_ch = row[7]
    # get the test section data from the pitot tube measured wrt p_atm
    datPt.p_total_inf = row[104] + datPt.p_atm
    # INVALID DATAPOINT datPt.p_static_inf = row[117] + datPt.p_atm

    # Following data is given as pressure difference
    datPt.airfoil_top_p_taps = row[8:33:] + datPt.p_atm # P001-P025
    datPt.airfoil_bottom_p_taps = row[33:57:] + datPt.p_atm # P026-P049
    datPt.rake_total_p_taps = row[57:104:] + datPt.p_atm # P050-P096
    datPt.rake_static_p_taps = row[105:117:] + datPt.p_atm # P098-P109
    datPt.init() # initialize datapoint by computing same needed values
    datapoints.append(datPt)

# print(datapoints[0].rake_total_p_taps)
# for i in datapoints[0].rake_pos_taps_total_p:
#     print(datapoints[0].rake_total_p_func(i))
#dp.plot_CL_a_curve(datapoints)
for i in datapoints:
    i.plot_Velocity_Deficit()
    i.plot_static_pressure_Deficit()
    #print(i.get_D())

#ptot =

