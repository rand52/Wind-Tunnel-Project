import math as mt
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

import Tap_Pos_Data_Reader as tappos


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
        self.p_tot_inf: float = None  # Total barometric pressure in stag ch
        self.rho_st_ch: float = None  # measured rho is stag ch

        self.airfoil_top_p_taps: np.ndarray = None # P001-P025
        self.airfoil_bottom_p_taps: np.ndarray =None # P026-P049
        self.rake_static_p_taps: np.ndarray = None # P050-P096
        self.rake_total_p_taps: np.ndarray =None # P098-P109
        # P97 is for pitot tube static pressure and not connected
        # P110-P113 are for pitot tubes total pressure and not connected

        # frequently used values and others needed for the class
        self.rake_static_p: sc.interpolate.interp1d = None
        self.rake_total_p: sc.interpolate.interp1d = None
        self.V_inf: float = None
        self.q_inf: float = None
        self.p_inf: float = None
        self.rho: float = None

    def init(self):
        """Computes and saves frequently used values for the datapoint that are needed for the class to properly
        function"""
        # compute frequently used values, order matters
        self.rho = self.get_rho()
        self.q_inf = self.get_q_inf()
        self.p_inf = self.get_p_inf()
        self.V_inf = self.get_V_inf()

        # interpolate the rake pressures with wake position
        self.rake_static_p = sc.interpolate.interp1d(
            self.rake_pos_taps_static_p,
            self.rake_static_p_taps,
            kind='linear',
            fill_value="extrapolate"
        )
        self.rake_total_p = sc.interpolate.interp1d(
            self.rake_pos_taps_total_p,
            self.rake_total_p_taps,
            kind='linear',
            fill_value="extrapolate"
        )

    def get_rho(self):
        R = 287.052874  # dry air
        T = self.temp_C + 273.15  # K
        return self.p_tot_inf / (R * T)

    def get_mu(self):
        mu0 = 1.716 * 10 ** -5
        T = self.temp_C + 273.15  # K
        S = 110.4  # K
        return mu0 * ((T / 273.15) ** (3. / 2)) * ((273.15 + S) / (T + S))

    def get_q_inf(self):
        return 0.211804 + 1.928442 * self.del_pb + (1.879374 * 10 ** -4) * (self.del_pb ** 2)

    def get_p_inf(self):
        return self.p_tot_inf - self.q_inf

    def get_V_inf(self):
        return mt.sqrt((2 * self.q_inf) / self.rho)

    def get_Re_inf(self):
        return (self.rho * self.V_inf * self.chord) / self.get_mu()

    def V_after_wing(self, y):
        q_inf = self.rake_total_p(y) - self.rake_static_p(y)
        # print(self.rake_total_p(y))
        # print(self.rake_static_p(y))
        # print(q_inf)
        # print(self.get_V_inf())
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
            print(self.p_inf - self.rake_static_p(y))
            return self.p_inf - self.rake_static_p(y)

        return self.get_rho() * sc.integrate.quad(inertia_deficit1, y_start, y_end)[0] + \
            sc.integrate.quad(inertia_deficit2, y_start, y_end)[0]

    def get_Cd(self):
        S = self.chord * self.span
        return self.get_D() / (self.q_inf * S)

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
        # Labeling the plot
        plt.title("Pressure Distribution on Airfoil")
        plt.xlabel("Position along chord (x/c)[%]")
        plt.ylabel("Pressure [Pa]")
        plt.legend()
        # Make a grid in the background for better readability
        plt.grid(True, linestyle='--', alpha=0.6)
        # Display
        plt.show()

    def plot_Cp(self):
        # normalize pressure readings to pressure coefficients Cps
        airfoil_top_cps = (self.airfoil_top_p_taps - self.p_inf) / self.q_inf
        print(self.airfoil_top_p_taps)
        print(airfoil_top_cps)
        airfoil_bottom_cps = (self.airfoil_bottom_p_taps - self.p_inf) / self.q_inf
        # plot the Cps
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        # Plot top side pressures
        plt.plot(
            self.airfoil_pos_top_taps,
            airfoil_top_cps,
            color="red",
            marker='.',
            label='Top Side (Red)',
            markersize=8
        )
        # Plot bottom side pressures
        plt.plot(
            self.airfoil_pos_bottom_taps,
            airfoil_bottom_cps,
            color="green",
            marker='s',
            label='Bottom Side (Green)',
            markersize=4,
            linestyle='-'  # Ensure lines connect the points
        )
        # Labeling the plot
        plt.title("Pressure Distribution on Airfoil")
        plt.xlabel("Position along chord (x/c)[%]")
        plt.ylabel("Cp [-]")
        plt.legend()
        # Make a grid in the background for better readability
        plt.grid(True, linestyle='--', alpha=0.6)
        # Display
        plt.show()
