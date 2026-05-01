class Ess:
    """Represents a PV which belongs to a bus, including all values for PV"""

    def __init__(self, ess_values):
        ### configuration

        self.ess_cap_kWh = ess_values['max_e_mwh'] * 1000
        self.ess_p_max = ess_values['p_mw'] * 1000

        #################
        # episode / time step values
        ###############

        self.p_ess_t = None
        self.penalties_t = None
        self.soc_ess_t = None
        self.action_t = None

    def calculate_p_soc_ess(self):

        desired_p_ess = self.action_t * self.ess_p_max
        if self.action_t >= 0:  # Charging
            feasible_p_ess = self.calculate_max_charging_power_ess()
        else:  # Discharging
            feasible_p_ess = self.calculate_max_discharging_power_ess()

        self.p_ess_t = min(desired_p_ess, feasible_p_ess) if self.action_t >= 0 else max(desired_p_ess, feasible_p_ess)
        ess_new_soc = self.update_soc_ess()

        if (self.action_t >= 0 and desired_p_ess > feasible_p_ess) or (self.action_t < 0 and desired_p_ess < feasible_p_ess):
            self.penalties_t = (desired_p_ess - feasible_p_ess) ** 2  # Squared difference as penalty
        else:
            self.penalties_t = 0

        return self.p_ess_t, ess_new_soc, self.penalties_t

    def calculate_max_charging_power_ess(self):

        remaining_capacity_kWh = (1 - self.soc_ess_t) * self.ess_cap_kWh
        remaining_capacity_power_kw = remaining_capacity_kWh * 4  # Convert to kW for a 15-minute interval
        feasible_p_cs = min(self.ess_p_max, remaining_capacity_power_kw)
        return feasible_p_cs

    def calculate_max_discharging_power_ess(self):
        # Calculate the available capacity to be discharged in kWh
        available_capacity_kWh = self.soc_ess_t * self.ess_cap_kWh
        # Convert the available capacity to an equivalent power value for the 15-minute timestep
        available_capacity_power_kw = available_capacity_kWh * 4  # Convert to kW for a 15-minute interval
        # The maximum feasible discharging power for this timestep is the lesser of the station's maximum output capability
        # or the power equivalent of discharging the available battery capacity in this timestep
        feasible_p_ess = min(self.ess_p_max, available_capacity_power_kw)
        # Ensure discharging does not result in negative SOC
        feasible_p_ess = max(-feasible_p_ess, -self.soc_ess_t * self.ess_cap_kWh * 4)
        return feasible_p_ess

    def update_soc_ess(self):
        # The division by 4 accounts for the conversion from power (kW) to energy (kWh) over a 15-minute interval
        new_soc = self.soc_ess_t + self.p_ess_t / 4 / self.ess_cap_kWh
        # Ensuring the SOC stays within the 0 to 1 range and rounding to 4 decimal places for precision
        new_soc = round(max(0, min(new_soc, 1)), 4)

        return new_soc
