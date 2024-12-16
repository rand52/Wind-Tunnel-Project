import os
import math as mt
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

import twoD_Tap_Pos_Data_Reader as tappos

# integration quality parameter
int_sub_div_lim: int = 100  # Increase the maximum sc.integrate integration quality
int_error_tolerance = 1e-6  # Increases sc.integrate integration quality by making it take more samples

#plotting constants
plt_line_width = 1
plt_circle_marker_size = 8
plt_square_marker_size = 4
plt_text_font_size = 10
plt_axis_font_size = 12
plt_legend_font_size = 10
plt_save_directory = "Plots"
plt_save_pad_inches = 0.1


class twoD_DP:
    chord: float = 0.16  # m  constant for test
    span: float = 0.4  # m constant for the test
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
        self.p_total_inf = None  # Total pressure in test section from pitot tube
        #INVALID DATAPOINT self.p_static_inf = None # Static pressure in test section from pitot tube
        self.rho_st_ch: float = None  # measured rho is stag ch

        self.airfoil_top_p_taps: np.ndarray = None  # P001-P025
        self.airfoil_bottom_p_taps: np.ndarray = None  # P026-P049
        self.rake_static_p_taps: np.ndarray = None  # P050-P096
        self.rake_total_p_taps: np.ndarray = None  # P098-P109

        # frequently used values and others needed for the class
        self.V_inf: float = None
        self.q_inf: float = None
        self.p_inf: float = None
        self.rho: float = None

        self.rake_static_p_func: sc.interpolate.interp1d = None
        self.rake_total_p_func: sc.interpolate.interp1d = None

        self.airfoil_top_taps_cps: np.ndarray = None
        self.airfoil_bottom_taps_cps: np.ndarray = None

        self.airfoil_top_cps_func: sc.interpolate.interp1d = None  # function normalized coordinates from 0-1 instead of 0-100%
        self.airfoil_bottom_cps_func: sc.interpolate.interp1d = None  # function normalized coordinates from 0-1 instead of 0-100%

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
        self.rake_static_p_func = sc.interpolate.InterpolatedUnivariateSpline(
            self.rake_pos_taps_static_p,
            self.rake_static_p_taps,
            k=2,  # Quadratic interpolation (degree 2)
            ext=3  # Constant extrapolation at the boundaries
            # VERY IMPORTANT for the static p to keep end values when
            # extrapolating otherwise polynomial fit gies big deviation
        )
        self.rake_total_p_func = sc.interpolate.InterpolatedUnivariateSpline(
            self.rake_pos_taps_total_p,
            self.rake_total_p_taps,
            k=2,  # Quadratic interpolation (degree 2)
            ext=3  # Constant extrapolation at the boundaries
        )
        self.airfoil_top_cps_func = sc.interpolate.interp1d(
            self.airfoil_pos_top_taps / 100,  # normalized coordinates from 0-1 instead of 0-100%
            self.airfoil_top_taps_cps,
            kind='linear',
            fill_value="extrapolate"
        )
        self.airfoil_bottom_cps_func = sc.interpolate.interp1d(
            self.airfoil_pos_bottom_taps / 100,  # normalized coordinates from 0-1 instead of 0-100%
            self.airfoil_bottom_taps_cps,
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

    def get_D(self, mode: str = "rake", neg_noise_reduction: bool = True):
        """mode=rake used for drag data from pressure rake
        mode=surface used for drag from  tap readings"""
        if mode == "rake":
            # integration param
            y_start = 0.0435
            y_end = max(self.rake_pos_taps_total_p) - 0.0435

            # internal intermediate functions for the integrands
            # also act as wrappers to handle the self instance, so expected type of func is passed to sc.integrate
            def V_at_rake_pos(y):
                q_inf_at_rake = self.rake_total_p_func(y) - self.rake_static_p_func(y)
                V_inf_at_rake = np.sqrt((2 * q_inf_at_rake) / self.rho)
                return V_inf_at_rake

            def velocity_deficit(y):
                v_deficit = self.V_inf - V_at_rake_pos(y)
                if neg_noise_reduction and v_deficit < 0:
                    v_deficit = 0  # set negative values to 0
                return v_deficit

            def inertia_deficit1(y):
                return V_at_rake_pos(y) * velocity_deficit(y)

            def inertia_deficit2(y):
                return self.p_inf - self.rake_static_p_func(y)

            # use integrate.quadrature instead of .quad as it's better at detecting narrow peaks
            return self.get_rho() * \
                sc.integrate.quad(inertia_deficit1, y_start, y_end, limit=int_sub_div_lim, epsabs=int_error_tolerance,
                                  epsrel=int_error_tolerance)[0]  #+ \
            #sc.integrate.quad(inertia_deficit2, y_start, y_end, limit=int_sub_div_lim,epsabs=int_error_tolerance,epsrel=int_error_tolerance)[0]
        elif mode == "surface":
            aoa_rad = mt.radians(self.aoa)
            return self.get_Cn() * mt.sin(aoa_rad)

    def get_Cd(self, mode: str = "rake"):
        """mode=rake used for drag data from pressure rake
        mode=surface used for drag from  tap readings"""
        # multiply by b=1, as this is an infinite airfoil and this is per unit span
        S = self.chord * 1
        return self.get_D(mode) / (self.q_inf * S)

    def get_Cn(self):
        def CP_bottom_minus_top_dif(x):
            return self.airfoil_bottom_cps_func(x) - self.airfoil_top_cps_func(x)

        return sc.integrate.quad(CP_bottom_minus_top_dif, 0, 1, limit=int_sub_div_lim)[0]

    def get_Cm_LE(self):
        def del_moment_contribution(x):
            return x * (self.airfoil_top_cps_func(x) - self.airfoil_bottom_cps_func(x))

        return sc.integrate.quad(del_moment_contribution, 0, 1, limit=int_sub_div_lim)[0]

    def get_Xcp(self):
        return -self.get_Cm_LE() / self.get_Cn()

    def get_Cm_quart_chord(self):
        return self.get_Cm_LE() + 0.25 * self.get_Cn()

    def get_Cl(self, mode: str = "rake"):
        """mode=rake used for drag data from pressure rake
        made=surface used for drag from  tap readings"""
        aoa_rad = mt.radians(self.aoa)
        if mode == "rake":
            return self.get_Cn() * (
                    mt.cos(aoa_rad) + (mt.sin(aoa_rad) ** 2) / mt.cos(aoa_rad)) - self.get_Cd() * mt.tan(
                aoa_rad)
        elif mode == "surface":
            return self.get_Cn() * mt.cos(aoa_rad)

    def plot_pressures(self, save=False):
        """save = True/False to save or not"""
        plt.figure(figsize=(7.8, 6))
        # Plot top side pressures
        plt.plot(
            self.airfoil_pos_top_taps,
            self.airfoil_top_p_taps,
            color="red",
            marker='.',
            label='Upper Surface (Red)',
            linewidth=plt_line_width,
            markersize=plt_circle_marker_size
        )
        # Plot bottom side pressures
        plt.plot(
            self.airfoil_pos_bottom_taps,
            self.airfoil_bottom_p_taps,
            color="green",
            marker='s',
            label='Lower Surface (Green)',
            linewidth=plt_line_width,
            markersize=plt_square_marker_size,
            linestyle='-'  # Ensure lines connect the points
        )
        # Display the AOA in the bottom-left corner
        plt.text(
            0.98, 0.15,
            fr"$\alpha$ = {self.aoa:.2f}°",  # format in scientific notation
            transform=plt.gca().transAxes,
            fontsize=plt_text_font_size,
            horizontalalignment='right'
        )
        # Display the reynolds_number in the bottom-left corner
        plt.text(
            0.98, 0.10,
            f"Re = {self.get_Re_inf():.2e}",  # format in scientific notation
            transform=plt.gca().transAxes,
            fontsize=plt_text_font_size,
            horizontalalignment='right'
        )
        # Display the total pressure in the top-left corner below legend
        plt.text(
            0.98, 0.05,
            f"Total Pressure: {self.p_total_inf:.0f} Pa",
            transform=plt.gca().transAxes,
            fontsize=plt_text_font_size,
            horizontalalignment='right'
        )
        # Labeling the plot
        plt.xlabel("Position along chord (x/c) [%]", fontsize=plt_axis_font_size)
        plt.ylabel("Pressure [Pa]", fontsize=plt_axis_font_size)
        plt.legend(fontsize=plt_legend_font_size)
        # Make a grid in the background for better readability
        plt.grid(True, linestyle='--', alpha=0.6)
        if save:
            os.makedirs(plt_save_directory, exist_ok=True)  # Ensure the directory exists
            file_path = os.path.join(plt_save_directory, f"2D_Cp_AOA_{self.aoa}.png")
            plt.savefig(file_path, bbox_inches='tight', pad_inches=plt_save_pad_inches)
        # Display the plot, after saving it
        plt.show()
        plt.close()  # Close the figure to free memory

    def plot_Cp(self, save: bool = False):
        """save = True/False to save or not"""
        # plot the Cps
        plt.figure(figsize=(7.8, 6))
        # Plot top side pressures
        plt.plot(
            self.airfoil_pos_top_taps,
            self.airfoil_top_taps_cps,
            color="red",
            marker='.',
            label='Upper Surface (Red)',
            linewidth=plt_line_width,
            markersize=plt_circle_marker_size
        )
        # Plot bottom side pressures
        plt.plot(
            self.airfoil_pos_bottom_taps,
            self.airfoil_bottom_taps_cps,
            color="green",
            marker='s',
            label='Lower Surface (Green)',
            linewidth=plt_line_width,
            markersize=plt_square_marker_size,
            linestyle='-'  # Ensure lines connect the points
        )
        # Display the AOA in the top-left corner below legend
        plt.text(
            0.98, 0.85,
            fr"$\alpha$ = {self.aoa:.2f}°",  # format in scientific notation
            transform=plt.gca().transAxes,
            fontsize=plt_text_font_size,
            horizontalalignment='right'
        )
        # Display the minimum Cp in the top-left corner below legend
        plt.text(
            0.98, 0.80,
            f"$C_{{p,\\text{{min}}}}$ : {min(np.min(self.airfoil_top_taps_cps), np.min(self.airfoil_bottom_taps_cps)):.2f}",
            transform=plt.gca().transAxes,
            fontsize=plt_text_font_size,
            horizontalalignment='right'
        )
        # Display the maximum Cp in the top-left corner below legend
        plt.text(
            0.98, 0.75,
            f"$C_{{p,\\text{{max}}}}$ : {max(np.max(self.airfoil_top_taps_cps), np.max(self.airfoil_bottom_taps_cps)):.2f}",
            transform=plt.gca().transAxes,
            fontsize=plt_text_font_size,
            horizontalalignment='right'
        )
        # Display the reynolds_number in the top-left corner below legend
        plt.text(
            0.98, 0.70,
            f"Re = {self.get_Re_inf():.2e}",  # format in scientific notation
            transform=plt.gca().transAxes,
            fontsize=plt_text_font_size,
            horizontalalignment='right'
        )
        # Invert the y-axis as it's a Cp plot
        plt.gca().invert_yaxis()
        # Labeling the plot
        plt.xlabel("Position along chord (x/c) [%]", fontsize=plt_axis_font_size)
        plt.ylabel(r"$C_p$ [-]", fontsize=plt_axis_font_size)
        plt.legend(fontsize=plt_legend_font_size)
        # Make a grid in the background for better readability
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Reference line for Cp = 0
        plt.grid(True, linestyle='--', alpha=0.6)
        # Save the plot to the specified directory
        if save:
            os.makedirs(plt_save_directory, exist_ok=True)  # Ensure the directory exists
            file_path = os.path.join(plt_save_directory, f"2D_Cp_AOA_{self.aoa}.png")
            plt.savefig(file_path, bbox_inches='tight', pad_inches=plt_save_pad_inches)
        # Display the plot, after saving it
        plt.show()
        plt.close()  # Close the figure to free memory

    def plot_Velocity_Deficit(self, mode: str = "fraction", neg_noise_reduction: bool = True, save: bool = False):
        """mode=fraction gives deficit as fraction of Vinf
        mode=actual gives deficit as actual velocity difference in m/s
        neg_noise_reduction=True removes the negative values
        save = True/False to save or not"""
        V_deficit = []
        for y in self.rake_pos_taps_total_p:
            q_inf_at_rake = self.rake_total_p_func(y) - self.rake_static_p_func(y)
            V_inf_at_rake = mt.sqrt((2 * q_inf_at_rake) / self.rho)

            # save different data depending on plotting case and take care of the noise reduction
            if mode == "fraction":
                if neg_noise_reduction and self.V_inf - V_inf_at_rake < 0:
                    V_deficit.append(1)  # set negative values of deficit to 0
                else:
                    V_deficit.append(V_inf_at_rake / self.V_inf)
            elif mode == "actual":
                if neg_noise_reduction and self.V_inf - V_inf_at_rake < 0:
                    V_deficit.append(0)  # set negative values of deficit to 0
                else:
                    V_deficit.append(self.V_inf - V_inf_at_rake)

        # plot the velocity deficit
        plt.figure(figsize=(7.8, 6))
        plt.plot(
            self.rake_pos_taps_total_p,
            V_deficit,
            color="blue",
            marker='s',
            label='Velocity deficit (blue)',
            linewidth=plt_line_width,
            markersize=plt_square_marker_size,
        )
        # Display the AOA in the bottom-left corner
        txtpos = [0.10, 0.05]
        if mode == "fraction":
            txtpos = [0.10, 0.05]
        elif mode == "actual":
            txtpos = [0.90, 0.85]
        plt.text(
            0.98, txtpos[0],
            fr"$\alpha$ = {self.aoa:.2f}°",  # format in scientific notation
            transform=plt.gca().transAxes,
            fontsize=plt_text_font_size,
            horizontalalignment='right'
        )
        # Display the reynolds_number in the bottom-left corner
        plt.text(
            0.98, txtpos[1],
            f"Re = {self.get_Re_inf():.2e}",  # format in scientific notation
            transform=plt.gca().transAxes,
            fontsize=plt_text_font_size,
            horizontalalignment='right'
        )
        # label the plot
        plt.legend(fontsize=plt_legend_font_size)
        plt.xlabel("Position along the rake [m]", fontsize=plt_axis_font_size)
        if mode == "fraction":
            plt.ylabel(r"Velocity at rake $u_1$ as a fraction of $V_{inf}$", fontsize=plt_axis_font_size)
        elif mode == "actual":
            plt.ylabel("Velocity deficit [m/s] at rake", fontsize=plt_axis_font_size)
        # Make a grid in the background for better readability
        if mode == "fraction":
            plt.axhline(1, color='black', linewidth=0.8, linestyle='--')  # Reference line
        elif mode == "actual":
            plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Reference line
        # grid
        plt.grid(True, linestyle='--', alpha=0.6)
        # saving
        if save:
            path = f"2D_V_deficit_AOA{self.aoa}.png"
            if mode == "fraction" and neg_noise_reduction:
                path = f"2D_V_deficit_fraction_AOA{self.aoa}.png"
            elif mode == "fraction" and not neg_noise_reduction:
                path = f"2D_V_deficit_fraction_no_noise_reduction_AOA{self.aoa}.png"
            elif mode == "actual" and neg_noise_reduction:
                path = f"2D_V_deficit_actual_AOA{self.aoa}.png"
            elif mode == "actual" and not neg_noise_reduction:
                path = f"2D_V_deficit_actual_AOA_no_noise_reduction{self.aoa}.png"
            os.makedirs("Plots", exist_ok=True)  # Ensure the directory exists
            full_file_path = os.path.join(plt_save_directory, path)
            plt.savefig(full_file_path, bbox_inches='tight', pad_inches=plt_save_pad_inches)
        # Display the plot, after saving it
        plt.show()
        plt.close()  # Close the figure to free memory

    def plot_static_pressure_Deficit(self, save: bool = False):
        """save = True/False to save or not"""
        p_static_deficit = []
        for y in self.rake_pos_taps_static_p:
            p_static_deficit.append(self.p_inf - self.rake_static_p_func(y))

        # plot the velocity deficit
        plt.figure(figsize=(7.8, 6))
        plt.plot(
            self.rake_pos_taps_static_p,
            p_static_deficit,
            color="blue",
            marker='s',
            markersize=plt_square_marker_size,
            label="Static p deficit (blue)"
        )
        # Display the AOA in the bottom-left corner
        plt.text(
            0.98, 0.10,
            fr"$\alpha$ = {self.aoa:.2f}°",  # format in scientific notation
            transform=plt.gca().transAxes,
            fontsize=plt_text_font_size,
            horizontalalignment='right'
        )
        # Display the reynolds_number in the bottom-left corner
        plt.text(
            0.98, 0.05,
            f"Re = {self.get_Re_inf():.2e}",  # format in scientific notation
            transform=plt.gca().transAxes,
            fontsize=plt_text_font_size,
            horizontalalignment='right'
        )
        plt.legend(fontsize=plt_legend_font_size)
        plt.xlabel("Position along the rake [m]", fontsize=plt_axis_font_size)
        plt.ylabel("Pressure deficit [Pa]", fontsize=plt_axis_font_size)
        # Make a grid in the background for better readability
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Reference line for Cp = 0
        plt.grid(True, linestyle='--', alpha=0.6)
        # saving
        if save:
            os.makedirs(plt_save_directory, exist_ok=True)  # Ensure the directory exists
            file_path = os.path.join(plt_save_directory, f"2D_static_p_deficit_AOA_{self.aoa}.png")
            plt.savefig(file_path, bbox_inches='tight', pad_inches=plt_save_pad_inches)
        # Display the plot, after saving it
        plt.show()
        plt.close()  # Close the figure to free memory


#### Plotting methods for multiple datapoints ####
def plot_CL_AOA_curve(datapoints: list[twoD_DP], mode: str = "rake", save: bool = False):
    """mode=rake used for drag data from pressure rake
    made=surface used for drag from  tap readings
    save = True/False to save or not"""
    # get the data in arrays, use np_arrays for speed
    AOA_s = np.array([datPt.aoa for datPt in datapoints])
    Cl_s = np.array([datPt.get_Cl(mode) for datPt in datapoints])
    Re_num_s = np.array([datPt.get_Re_inf() for datPt in datapoints])

    # plot the Cl-a curve
    plt.figure(figsize=(7.8, 6))
    plt.plot(
        AOA_s,
        Cl_s,
        color="blue",
        marker='.',
        label = r"$\alpha$ vs $C_l$",
        markersize=plt_circle_marker_size
    )
    # Display the maximum CL in the bottom-right corner
    plt.text(
        0.98, 0.15,
        f"Cl at stall: {np.max(Cl_s):.2f}",
        transform=plt.gca().transAxes,
        fontsize=plt_text_font_size,
        horizontalalignment='right'
    )
    # Display the AOA at the maximum CL
    plt.text(
        0.98, 0.10,
        f"AOA at stall: {AOA_s[np.argmax(Cl_s)]:.2f}",
        transform=plt.gca().transAxes,
        fontsize=plt_text_font_size,
        horizontalalignment='right'
    )
    # Display the average reynolds number in the bottom-right corner
    Re_avg = sum(Re_num_s) / len(Re_num_s)
    plt.text(
        0.98, 0.05,
        f"Re = {Re_avg:.2e}",  # format in scientific notation
        transform=plt.gca().transAxes,
        fontsize=plt_text_font_size,
        horizontalalignment='right'
    )
    plt.legend(fontsize = plt_legend_font_size)
    # Labeling the plot
    plt.xlabel(r"$\alpha$ [°]", fontsize=plt_axis_font_size)
    plt.ylabel(r"$C_l$ [-]", fontsize=plt_axis_font_size)
    # Make a grid in the background for better readability
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Reference line for Cl = 0
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')  # Reference line for AOA = 0deg
    # Major grid
    plt.grid(True, linestyle='--', alpha=0.6)
    # Saving
    if save:
        path = f"2D_Cl-alpha_plot.png"
        if mode == "surface":
            path = f"2D_Cl-alpha_plot_from_SURFACE_reading_only.png"
        os.makedirs(plt_save_directory, exist_ok=True)  # Ensure the directory exists
        full_file_path = os.path.join(plt_save_directory, path)
        plt.savefig(full_file_path, bbox_inches='tight', pad_inches=plt_save_pad_inches)
    # Display the plot, after saving it
    plt.show()
    plt.close()  # Close the figure to free memory


def plot_drag_polar(datapoints: list[twoD_DP], mode: str = "rake", save: bool = False):
    """mode=rake used for drag data from pressure rake
    made=surface used for drag from  tap readings
    save = True/False to save or not"""
    # get the data in arrays, use np_arrays for speed
    Cl_s = np.array([datPt.get_Cl(mode) for datPt in datapoints])
    Cd_s = np.array([datPt.get_Cd(mode) for datPt in datapoints])
    Re_num_s = np.array([datPt.get_Re_inf() for datPt in datapoints])

    # plot the Cl-Cd curve
    plt.figure(figsize=(7.8, 6))
    plt.plot(
        Cd_s,
        Cl_s,
        color="blue",
        marker='.',
        markersize=plt_circle_marker_size,
        label = r"$C_l$ vs $C_d$"
    )
    # Display the average reynolds number in the bottom-right corner
    Re_avg = sum(Re_num_s) / len(Re_num_s)
    plt.text(
        0.98, 0.05,
        f"Re = {Re_avg:.2e}",  # format in scientific notation
        transform=plt.gca().transAxes,
        fontsize=plt_text_font_size,
        horizontalalignment='right'
    )
    plt.legend(fontsize=plt_legend_font_size)
    # Labeling the plot
    plt.xlabel(r"$C_d$ [-]", fontsize=plt_axis_font_size)
    plt.ylabel(r"$C_l$ [-]", fontsize=plt_axis_font_size)
    # Major grid
    plt.grid(True, linestyle='--', color='gray', alpha=0.6)
    # Minor grid
    plt.minorticks_on()  # Enable minor grid
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.002))  # Minor grid spacing for x-axis
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.05))  # Minor grid spacing for y-axis
    plt.grid(True, linestyle=':', color='gray', linewidth=0.5, alpha=0.5, which='minor', axis='both')
    # saving
    if save:
        path = f"2D_Cl-alpha_plot.png"
        if mode == "surface":
            path = f"2D_drag_polar_from_SURFACE_readings_only.png"
        os.makedirs(plt_save_directory, exist_ok=True)  # Ensure the directory exists
        full_file_path = os.path.join(plt_save_directory, path)
        plt.savefig(full_file_path, bbox_inches='tight', pad_inches=plt_save_pad_inches)
    # Display the plot, after saving it
    plt.show()
    plt.close()  # Close the figure to free memory


def plot_Cm_AOA_curve(datapoints: list[twoD_DP], mode="quarter", save: bool = False):
    """mode = quarter for quarter chord moment
    mode = le for leading edge moment
    save = True/False to save or not"""
    # get the data in arrays, use np_arrays for speed
    AOA_s = np.array([datPt.aoa for datPt in datapoints])
    Re_num_s = np.array([datPt.get_Re_inf() for datPt in datapoints])
    Cm_s = np.empty(0)
    label = r"$\alpha$ vs $C_{m_{c/4}}$"
    txt_pos = [0.02, 0.11]
    txt_align = "left"
    if mode == "quarter":
        Cm_s = np.array([datPt.get_Cm_quart_chord() for datPt in datapoints])
        label=r"$\alpha$ vs $C_{m_{c/4}}$"
        txt_pos = [0.02, 0.11]
        txt_align = "left"
    elif mode == "le":
        Cm_s = np.array([datPt.get_Cm_LE() for datPt in datapoints])
        label=r"$\alpha$ vs $C_{m_{LE}}$"
        txt_pos = [0.98, 0.85]
        txt_align = "right"

    # plot the Cl-a curve
    plt.figure(figsize=(7.8, 6))
    plt.plot(
        AOA_s,
        Cm_s,
        color="blue",
        marker='.',
        markersize=plt_circle_marker_size,
        label = label
    )
    # Display the average reynolds number in the bottom-left corner
    Re_avg = sum(Re_num_s) / len(Re_num_s)
    plt.text(
        txt_pos[0], txt_pos[1],
        f"Re = {Re_avg:.2e}",  # format in scientific notation
        transform=plt.gca().transAxes,
        fontsize=plt_text_font_size,
        horizontalalignment=txt_align
    )
    plt.legend(fontsize=plt_legend_font_size)
    # Labeling the plot
    plt.xlabel(r"$\alpha$ [°]", fontsize=plt_axis_font_size)
    if mode == "quarter":
        plt.ylabel(r"$C_{m_{c/4}}$ [-]", fontsize=plt_axis_font_size)
    elif mode == "le":
        plt.ylabel(r"$C_{m_{LE}}$ [-]", fontsize=plt_axis_font_size)
    # Make a grid in the background for better readability
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Reference line for AOA = 0deg
    # Major grid
    plt.grid(True, linestyle='--', alpha=0.6)
    if save:
        path = f"2D_Cm-alpha_plot"
        if mode == "quarter":
            path = f"2D_Cm_quarter_c-alpha_plot"
        elif mode == "le":
            path = f"2D_Cm_LE-alpha_plot"
        os.makedirs(plt_save_directory, exist_ok=True)  # Ensure the directory exists
        full_file_path = os.path.join(plt_save_directory, path)
        plt.savefig(full_file_path, bbox_inches='tight', pad_inches=plt_save_pad_inches)
    # Display the plot, after saving it
    plt.show()
    plt.close()  # Close the figure to free memory


def plot_Xcp_AOA_curve(datapoints: list[twoD_DP], save: bool = False):
    """save = True/False to save or not"""
    # get the data in arrays, use np_arrays for speed
    AOA_s = np.array([datPt.aoa for datPt in datapoints])
    Xcp_s = np.array([datPt.get_Xcp() for datPt in datapoints])
    Re_num_s = np.array([datPt.get_Re_inf() for datPt in datapoints])

    # plot the Cl-a curve
    plt.figure(figsize=(7.8, 6))
    plt.plot(
        AOA_s,
        Xcp_s,
        color="blue",
        marker='.',
        label=r"$\alpha$ vs $X_{Cp}$",
        markersize=plt_circle_marker_size
    )
    # Display the average reynolds number in the bottom-left corner
    Re_avg = sum(Re_num_s) / len(Re_num_s)
    plt.text(
        0.98, 0.89,
        f"Re = {Re_avg:.2e}",  # format in scientific notation
        transform=plt.gca().transAxes,
        fontsize=plt_text_font_size,
        horizontalalignment='right'
    )
    # Labeling the plot
    plt.legend(fontsize=plt_legend_font_size)
    plt.xlabel(r"$\alpha$ [°]", fontsize=plt_axis_font_size)
    plt.ylabel(r"$X_{Cp}$ [x/c]", fontsize=plt_axis_font_size)
    # Make a grid in the background for better readability
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Reference line for AOA = 0deg
    # Major grid
    plt.grid(True, linestyle='--', alpha=0.6)
    # saving
    if save:
        os.makedirs(plt_save_directory, exist_ok=True)  # Ensure the directory exists
        file_path = os.path.join(plt_save_directory, f"2D_Xcp-alpha_plot.png")
        plt.savefig(file_path, bbox_inches='tight', pad_inches=plt_save_pad_inches)
    # Display the plot, after saving it
    plt.show()
    plt.close()  # Close the figure to free memory
