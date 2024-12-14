import numpy as np
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
    datPt.p_atm = row[4] * 100  # it's given in kPa
    datPt.temp_C = row[5]
    datPt.rho_st_ch = row[7]
    # get the test section data from the pitot tube measured wrt p_atm
    datPt.p_total_inf = row[104] + datPt.p_atm
    datPt.p_static_inf = row[117] + datPt.p_atm
    # Following data is given as pressure difference
    datPt.airfoil_top_p_taps = row[8:33:] + datPt.p_atm # P001-P025
    datPt.airfoil_bottom_p_taps = row[33:57:] + datPt.p_atm # P026-P049
    datPt.rake_total_p_taps = row[57:104:] + datPt.p_atm # P050-P096
    datPt.rake_static_p_taps = row[105:117:] + datPt.p_atm # P098-P109
    datPt.init() # initialize datapoint by computing same needed values
    datapoints.append(datPt)

# for i in np.arange(0, 0.225, 0.001):
#     print(datapoints[10].V_inf,"  ",datapoints[10].V_after_wing(i)," diff ",datapoints[10].V_inf-datapoints[10].V_after_wing(i))
# for i in datapoints:
#     print(i.get_D())
#print(datapoints[10].get_D())
# datapoints[10].plot_pressures()
print(datapoints[3].aoa)
datapoints[3].plot_Cp()
#print(datapoints[0].rake_static_p_taps)
