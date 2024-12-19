from twoD_Data_Reader import datapoints
import twoD_Datapoint as dp

# uncomment what you want to plot
# datapoints go by array number if you need a particular one you can do for example
# datapoints[3].plot_Cp()
# add save=True for graph to be saved in a plots folder in the same directory (or change from twoD_Datapoint)

# for i in datapoints:
#     i.plot_rake_velocity_profile(save=True)
#     i.plot_pressures() # pressures on airfoil
#     i.plot_Cp() # Cps on airfoil
#     i.plot_Velocity_Deficit() # velocity deficit faction
#     i.plot_Velocity_Deficit(mode="fraction",neg_noise_reduction=False) # same with no noise reduction
#     i.plot_Velocity_Deficit(mode="actual") # actual deficit in m/s
#     i.plot_Velocity_Deficit(mode="actual",neg_noise_reduction=False) # same with no noise reduction
#     i.plot_static_pressure_Deficit() # static pressure deficit

# dp.plot_multiple_velocity_profiles(datapoints[13:39:12]) # multiple velocity profiles

# dp.plot_CL_AOA_curve(datapoints[0:39:],mode="surface") # Cl-a plot form surface taps only
# dp.plot_CL_AOA_curve(datapoints[0:39:]) # Cl-a plot actual
# dp.plot_CL_AOA_curve(datapoints[0:39:],mode="compare") # compare from wake rake and from pressure taps
# dp.plot_CL_AOA_curve(datapoints,color_split= 39) # Cl-a plot hysteresis values included
# dp.plot_CL_AOA_curve(datapoints,color_split= 39,mode="compare") # Cl-a plot hysteresis values included and comparison between rake and surface data

# dp.plot_drag_polar(datapoints[0:39:], mode="surface") # Cd-Cl plot form surface taps only
# dp.plot_drag_polar(datapoints[0:39:],save=True) # Cd-Cl plot actual
# dp.plot_drag_polar(datapoints[0:39:], mode="compare") # compare from wake rake and from pressure taps
# dp.plot_drag_polar(datapoints,color_split= 39) # Cd-Cl plot hysteresis values included
# dp.plot_drag_polar(datapoints,color_split= 39, mode="compare") # Cd-Cl plot hysteresis values included and comparison between rake and surface data

# dp.plot_Xcp_AOA_curve(datapoints[0:39:]) # X_cp vs aoa plot
# dp.plot_Cm_AOA_curve(datapoints[0:39:],mode="quarter") #Cm_c/4 vs aoa plot
# dp.plot_Cm_AOA_curve(datapoints[0:39:],mode="le") #Cm_LE vs aoa plot