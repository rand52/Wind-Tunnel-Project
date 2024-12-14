import pandas as pd
import scipy as sc

import twoD_Datapoint as dp

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
    datPt.p_tot_inf = row[4] * 100  # it's given in kPa
    datPt.temp_C = row[5]
    datPt.rho_st_ch = row[7]
    datPt.airfoil_top_p_taps = row[8:33:]  # P001-P025
    datPt.airfoil_bottom_p_taps = row[33:57:]   # P026-P049
    datPt.rake_total_p_taps = row[57:104:]   # P050-P096
    datPt.rake_static_p_taps = row[105:117:]   # P098-P109
    datapoints.append(datPt)


#datapoints[0].plot_pressures()
print(datapoints[0].rake_static_p_taps)
