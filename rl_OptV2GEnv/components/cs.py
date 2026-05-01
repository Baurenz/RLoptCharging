class CS:
    """Represents a PV which belongs to a bus, including all values for PV"""

    def __init__(self, config_cs):
        ### configuration
        config_cs['max_e_mwh']

        self.cs_cap_kWh = config_cs['max_e_mwh'] * 1000
        self.cs_p_max = config_cs['p_mw'] * 1000
        self.v2g = config_cs['v2g']

        #################
        # episode / time step values
        ###############
        self.action_t = None
        self.ev_present_t = None
        self.desired_p_cs = None
        self.soc_cs_t = None
        self.soc_cs_next_t = None
        self.p_cs_t = None
        self.penalty_os_t = None

    def calculate_max_charging_power_cs(self):

        remaining_capacity_kWh = (1 - self.soc_cs_t) * self.cs_cap_kWh
        remaining_capacity_power_kw = remaining_capacity_kWh * 4  # Convert to kW for a 15-minute interval
        feasible_p_cs = min(self.cs_p_max, remaining_capacity_power_kw)
        return feasible_p_cs

    def calculate_max_discharging_power_cs(self):
        # Calculate the available capacity to be discharged in kWh
        available_capacity_kWh = self.soc_cs_t * self.cs_cap_kWh
        # Convert the available capacity to an equivalent power value for the 15-minute timestep
        available_capacity_power_kw = available_capacity_kWh * 4  # Convert to kW for a 15-minute interval
        # The maximum feasible discharging power for this timestep is the lesser of the station's maximum output capability
        # or the power equivalent of discharging the available battery capacity in this timestep
        feasible_p_cs = min(self.cs_p_max, available_capacity_power_kw)
        # Ensure discharging does not result in negative SOC
        feasible_p_cs = max(-feasible_p_cs, -self.soc_cs_t * self.cs_cap_kWh * 4)
        return feasible_p_cs

    def update_soc_cs(self):
        # Calculate the new SOC, ensuring the result is for a 15-minute timestep
        new_soc = self.soc_cs_t + self.p_cs_t / 4 / self.cs_cap_kWh
        new_soc = round(new_soc, 4)
        # Update SOC for the next timestep

        return new_soc
    
    def calculate_p_soc_cs(self):

        # if cs.v2g = false due to symmetric action space [-1,1] action needs to be converted to appropriate range
        if not self.v2g:
            self.action_t = (self.action_t + 1) / 2

        if self.ev_present_t:  # Check if a car is present
            desired_p_cs = self.action_t * self.cs_p_max
            if self.action_t >= 0:
                feasible_p_cs = self.calculate_max_charging_power_cs()
            else:
                feasible_p_cs = self.calculate_max_discharging_power_cs()

            self.p_cs_t = min(desired_p_cs, feasible_p_cs) if self.action_t >= 0 else max(desired_p_cs, feasible_p_cs)
            p_cs = self.p_cs_t

            cs_new_soc = self.update_soc_cs()

            if (self.action_t >= 0 and desired_p_cs > feasible_p_cs) or (
                    self.action_t < 0 and desired_p_cs < feasible_p_cs):
                penalty_cs = (desired_p_cs - feasible_p_cs) ** 2  # Squared difference as penalty
            else:
                penalty_cs = 0

            self.penalty_os_t = penalty_cs

        else:
            self.p_cs_t = 0
            p_cs = 0
            penalty_cs = 0
            cs_new_soc = self.soc_cs_next_t
            self.penalty_os_t = penalty_cs

        return p_cs, cs_new_soc, penalty_cs
