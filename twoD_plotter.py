from twoD_Data_Reader import datapoints
import twoD_Datapoint as dp

# for i in datapoints:
#      i.plot_pressures()
#      i.plot_Cp()
#      i.plot_Velocity_Deficit()
#      i.plot_Velocity_Deficit(mode="fraction",neg_noise_reduction=False)
#      i.plot_Velocity_Deficit(mode="actual")
#      i.plot_Velocity_Deficit(mode="actual",neg_noise_reduction=False)
#      i.plot_static_pressure_Deficit()
# dp.plot_CL_AOA_curve(datapoints[0:39:],mode="surface") # Cl-a plot form surface taps only
# dp.plot_CL_AOA_curve(datapoints[0:39:]) # Cl-a plot actual
# dp.plot_CL_AOA_curve(datapoints,color_split= 39) # # Cl-a plot hysteresis values included

dp.plot_drag_polar(datapoints[0:39:], mode="surface") # Cd-Cl plot form surface taps only
# dp.plot_drag_polar(datapoints[0:39:]) # Cd-Cl plot actual
# dp.plot_drag_polar(datapoints,color_split= 39) # Cd-Cl plot hysteresis values included

# dp.plot_Xcp_AOA_curve(datapoints[0:39:]) # X_cp vs aoa plot
# dp.plot_Cm_AOA_curve(datapoints[0:39:],mode="quarter") #Cm_c/4 vs aoa plot
# dp.plot_Cm_AOA_curve(datapoints[0:39:],mode="le") #Cm_LE vs aoa plot