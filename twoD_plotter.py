from twoD_Data_Reader import datapoints
import twoD_Datapoint as dp

for i in datapoints:
#     i.plot_pressures()
    i.plot_Cp()
#     i.plot_Velocity_Deficit()
#     i.plot_Velocity_Deficit(mode="fraction",neg_noise_reduction=False)

#dp.plot_CL_AOA_curve(datapoints[0:39:])
#dp.plot_drag_polar(datapoints[0:39:])
#dp.plot_Xcp_AOA_curve(datapoints[0:39:])
#dp.plot_Cm_AOA_curve(datapoints[0:39:],mode="le")