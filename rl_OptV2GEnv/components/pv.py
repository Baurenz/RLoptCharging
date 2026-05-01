import numpy as np


class PV:
    """Represents a PV which belongs to a bus, including all values for PV"""

    def __init__(self, kWp):

        self.kWp = kWp
        self.pv_prod_episode = []


        # Constants for calculating production from ambient data
        self.G_standard = 1000  # Standard solar radiation in W/m²
        self.T_standard = 25  # Standard cell temperature in °C
        self.alpha_T = -0.004  # Temperature coefficient in 1/°C

    def get_pv_prod_ep_ambient(self, irr_temp_cloud_ep, bus_id, ep_count):

        rng = np.random.RandomState(bus_id + ep_count)

        pv_kWp = self.kWp  # Rated power of the PV system in kWp
        # Initialize an array to hold the PV production values for each timestep
        pv_prod = np.zeros(len(irr_temp_cloud_ep))
        for i, (G, T_ambient, cloud_opacity) in enumerate(irr_temp_cloud_ep):
            # Calculate cell temperature for the current timestep
            T_cell = T_ambient + (self.T_standard / 800) * G

            # Calculate PV power output for the current timestep
            p_pv = pv_kWp * (G / self.G_standard) * (1 + self.alpha_T * (T_cell - self.T_standard))
            # Ensure that PV power output is not negative
            p_pv = max(p_pv, 0)

            # Determine the noise level based on cloud opacity
            if 30 <= cloud_opacity <= 70:
                noise_level = -abs(rng.normal(0, 0.1))  # Higher variability in this cloud opacity range
            else:
                noise_level = -abs(rng.normal(0, 0.05))  # Lower variability outside this range

            # Apply noise level to the calculated PV power output
            p_pv_noise = p_pv * (1 + noise_level)
            p_pv_noise = max(p_pv_noise, 0)  # Ensure that PV power output with noise is not negative
            pv_prod[i] = p_pv_noise

        self.pv_prod_episode = pv_prod
