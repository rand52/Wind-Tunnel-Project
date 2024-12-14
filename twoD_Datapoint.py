import math as mt
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

import Tap_Pos_Data_Reader as tappos

# integration quality parameter
int_sub_div_lim: int = 100  # Increase the maximum sc.integrate method subdivisions for integration


class twoD_DP:
    chord: float = 0.16  #m  constant for test
    span: float = 0.4  #m constant for the test
    rake_pos_taps_total_p = tappos.rake_pos_taps_total_p
    rake_pos_taps_static_p = tappos.rake_pos_taps_static_p
    airfoil_pos_top_taps = tappos.airfoil_pos_top_taps
    airfoil_pos_bottom_taps = tappos.airfoil_pos_bottom_taps

    def __init__(self):
        """Constructor for the 2D datapoint class. Takes in all measured data for a certain AOA["""
        self.aoa: float = None  # AOA
        self.temp_C: float = None  # Temp in C
        self.del_pb: float = None  # Pressure dif in stag ch and contraction
        self.p_atm: float = None  # Atmospheric barometric pressure used as a datum
        # P97 is the total pressure measured in wind tunnel by pitot tube and is accurate
        # P110 is static pressure measured by the pitot tube which is disturbed and isn't accurate and shouldn't be used
        self.p_total_inf = None # Total pressure in test section from pitot tube
        #INVALID DATAPOINT self.p_static_inf = None # Static pressure in test section from pitot tube
        self.rho_st_ch: float = None  # measured rho is stag ch

        self.airfoil_top_p_taps: np.ndarray = None # P001-P025
        self.airfoil_bottom_p_taps: np.ndarray =None # P026-P049
        self.rake_static_p_taps: np.ndarray = None # P050-P096
        self.rake_total_p_taps: np.ndarray =None # P098-P109

        # frequently used values and others needed for the class
        self.V_inf: float = None
        self.q_inf: float = None
        self.p_inf: float = None
        self.rho: float = None

        self.rake_static_p_func: sc.interpolate.interp1d = None
        self.rake_total_p_func: sc.interpolate.interp1d = None

        self.airfoil_top_taps_cps: np.ndarray = None
        self.airfoil_bottom_taps_cps: np.ndarray = None

        self.airfoil_top_cps_func: sc.interpolate.interp1d = None # function normalized coordinates from 0-1 instead of 0-100%
        self.airfoil_bottom_cps_func: sc.interpolate.interp1d = None # function normalized coordinates from 0-1 instead of 0-100%

    def init(self):
        """Computes and saves frequently used values for the datapoint that are needed for the class to properly
        function"""
        # compute frequently used values, order matters
        self.q_inf = self.get_q_inf()
        self.p_inf = self.get_p_inf()
        self.rho = self.get_rho()
        self.V_inf = self.get_V_inf()

        self.airfoil_top_taps_cps = (self.airfoil_top_p_taps - self.p_inf) / self.q_inf
        self.airfoil_bottom_taps_cps = (self.airfoil_bottom_p_taps - self.p_inf) / self.q_inf

        # interpolate the rake pressures with wake position
        self.rake_static_p_func = sc.interpolate.interp1d(
            self.rake_pos_taps_static_p,
            self.rake_static_p_taps,
            kind='linear',
            fill_value="extrapolate"
        )
        self.rake_total_p_func = sc.interpolate.interp1d(
            self.rake_pos_taps_total_p,
            self.rake_total_p_taps,
            kind='linear',
            fill_value="extrapolate"
        )
        self.airfoil_top_cps_func = sc.interpolate.interp1d(
            self.airfoil_top_taps_cps,
            self.airfoil_pos_top_taps/100, #normalized coordinates from 0-1 instead of 0-100%
            kind='linear',
            fill_value="extrapolate"
        )
        self.airfoil_bottom_cps_func = sc.interpolate.interp1d(
            self.airfoil_bottom_taps_cps,
            self.airfoil_pos_bottom_taps/100, #normalized coordinates from 0-1 instead of 0-100%
            kind='linear',
            fill_value="extrapolate"
        )

    def get_rho(self):
        R = 287.052874  # dry air
        T = self.temp_C + 273.15  # K
        return self.p_inf / (R * T)

    def get_mu(self):
        mu0 = 1.716 * 10 ** -5
        T = self.temp_C + 273.15  # K
        S = 110.4  # K
        return mu0 * ((T / 273.15) ** (3. / 2)) * ((273.15 + S) / (T + S))

    def get_q_inf(self):
        # don't use static pressure reading but best fit polynomial as it's more accurate
        return 0.211804 + 1.928442 * self.del_pb + (1.879374 * 10 ** -4) * (self.del_pb ** 2)

    def get_p_inf(self):
        # don't use static pressure reading but best fit polynomial as it's more accurate
        return self.p_total_inf - self.q_inf

    def get_V_inf(self):
        return mt.sqrt((2 * self.q_inf) / self.rho)

    def get_Re_inf(self):
        return (self.rho * self.V_inf * self.chord) / self.get_mu()

    def V_after_wing(self, y):
        q_inf = self.rake_total_p_func(y) - self.rake_static_p_func(y)
        return mt.sqrt((2 * q_inf) / self.rho)

    def get_D(self):
        # integration param
        y_start = 0
        y_end = max(self.rake_pos_taps_total_p)

        # internal intermediate functions for the integrands
        # also act as wrappers to handle the self instance, so expected type of func is passed to sc.integrate
        def inertia_deficit1(y):
            return self.V_after_wing(y) * (self.V_inf - self.V_after_wing(y))

        def inertia_deficit2(y):
            return self.p_inf - self.rake_static_p_func(y)

        return self.get_rho() * sc.integrate.quad(inertia_deficit1, y_start, y_end,limit=int_sub_div_lim)[0] + \
            sc.integrate.quad(inertia_deficit2, y_start, y_end,limit=int_sub_div_lim)[0]

    def get_Cd(self):
        S = self.chord * self.span
        return self.get_D() / (self.q_inf * S)

    def get_Cn(self):
        return 0

    def plot_pressures(self):
        plt.figure(figsize=(10, 6))
        # Plot top side pressures
        plt.plot(
            self.airfoil_pos_top_taps,
            self.airfoil_top_p_taps,
            color="red",
            marker='.',
            label='Top Side (Red)',
            markersize=8
        )
        # Plot bottom side pressures
        plt.plot(
            self.airfoil_pos_bottom_taps,
            self.airfoil_bottom_p_taps,
            color="green",
            marker='s',
            label='Bottom Side (Green)',
            markersize=4,
            linestyle='-'  # Ensure lines connect the points
        )
        # Display the total pressure in the top-left corner below legend
        plt.text(
            0.98, 0.85,
            f"Total Pressure: {self.p_total_inf:.0f} Pa",
            transform=plt.gca().transAxes,
            fontsize=10,
            horizontalalignment='right'
        )
        # Display the reynolds_number in the top-left corner below legend
        plt.text(
            0.98, 0.80,
            f"Reynolds Number: {self.get_Re_inf():.2e}", # format in scientific notation
            transform=plt.gca().transAxes,
            fontsize=10,
            horizontalalignment='right'
        )
        # Labeling the plot
        plt.title(f"Pressure Distribution on Airfoil AOA={self.aoa} deg")
        plt.xlabel("Position along chord (x/c)[%]")
        plt.ylabel("Pressure [Pa]")
        plt.legend()
        # Make a grid in the background for better readability
        plt.grid(True, linestyle='--', alpha=0.6)
        # Display
        plt.show()

    def plot_Cp(self):
        # plot the Cps
        plt.figure(figsize=(10, 6))
        # Plot top side pressures
        plt.plot(
            self.airfoil_pos_top_taps,
            self.airfoil_top_taps_cps,
            color="red",
            marker='.',
            label='Top Side (Red)',
            markersize=8
        )
        # Plot bottom side pressures
        plt.plot(
            self.airfoil_pos_bottom_taps,
            self.airfoil_bottom_taps_cps,
            color="green",
            marker='s',
            label='Bottom Side (Green)',
            markersize=4,
            linestyle='-'  # Ensure lines connect the points
        )
        # Display the minimum Cp in the top-left corner below legend
        plt.text(
            0.98, 0.85,
            f"Cp min: {min(np.min(self.airfoil_top_taps_cps),np.min(self.airfoil_bottom_taps_cps)):.2f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            horizontalalignment='right'
        )
        # Display the maximum Cp in the top-left corner below legend
        plt.text(
            0.98, 0.80,
            f"Cp max: {max(np.max(self.airfoil_top_taps_cps),np.max(self.airfoil_bottom_taps_cps)):.2f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            horizontalalignment='right'
        )
        # Display the reynolds_number in the top-left corner below legend
        plt.text(
            0.98, 0.75,
            f"Reynolds Number: {self.get_Re_inf():.2e}", # format in scientific notation
            transform=plt.gca().transAxes,
            fontsize=10,
            horizontalalignment='right'
        )
        # Invert the y-axis as it's a Cp plot
        plt.gca().invert_yaxis()
        # Labeling the plot
        plt.title(f"Pressure Distribution on Airfoil at AOA={self.aoa} deg")
        plt.xlabel("Position along chord (x/c)[%]")
        plt.ylabel("Cp [-]")
        plt.legend()
        # Make a grid in the background for better readability
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Reference line for Cp = 0
        plt.grid(True, linestyle='--', alpha=0.6)
        # Display
        plt.show()

    def plot_Velocity_Deficit(self) :
        V_deficit = []
        for i in range(len(self.rake_pos_taps_total_p)):
            q_inf_at_rake = self.rake_total_p_taps[i] - self.rake_static_p_func(self.rake_pos_taps_total_p[i])
            V_inf_at_rake = mt.sqrt((2 * q_inf_at_rake) / self.rho)
            V_deficit.append(V_inf_at_rake/self.V_inf)

        # plot the velocity deficit
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.rake_pos_taps_total_p,
            V_deficit,
            color="blue",
            marker='s',
            markersize=3
        )
        plt.title("velocity deficit behind airfoil trailing edge")
        plt.xlabel("Position along the rake [m]")
        plt.ylabel("Velocity deficit [%]")
        # Make a grid in the background for better readability
        plt.axhline(1, color='black', linewidth=0.8, linestyle='--')  # Reference line for Cp = 0
        plt.grid(True, linestyle='--', alpha=0.6)
        # Display
        plt.show()

    def plot_static_pressure_Deficit(self):
        p_static_deficit = []
        for i in range(len(self.rake_pos_taps_total_p)):
            p_static_deficit.append(self.p_inf - self.rake_static_p_func(self.rake_pos_taps_total_p[i]))

        # plot the velocity deficit
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.rake_pos_taps_total_p,
            p_static_deficit,
            color="blue",
            marker='s',
            markersize=3
        )
        plt.title("static pressure deficit behind airfoil trailing edge")
        plt.xlabel("Position along the rake [m]")
        plt.ylabel("Pressure deficit [Pa]")
        # Make a grid in the background for better readability
        plt.axhline(1, color='black', linewidth=0.8, linestyle='--')  # Reference line for Cp = 0
        plt.grid(True, linestyle='--', alpha=0.6)
        # Display
        plt.show()
