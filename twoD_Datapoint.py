
class twoD_DP:

    def __init__(self):
        """Constructor for the 2D datapoint class. Takes in all measured data for a certain AOA["""
        self.aoa: float = None
        self.p_atm : float = None
        self.rho : float = None
        self.temp_C : float = None # Temp in C
        self.del_pb : float = None # For V from pressure dif in stag ch and test sec
        self.p_stat : float = None # P97

        self.airfoil_top_p_taps = [] # P001-P025
        self.airfoil_bottom_p_taps = [] # P026-P049
        self.rake_static_p_taps = [] # P050-P096
        self.rake_total_p_taps = [] # P098-P109
        # P110-P113 are for pitot tubes and not connceted