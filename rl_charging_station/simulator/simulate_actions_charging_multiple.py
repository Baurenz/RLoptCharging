import numpy as np
from network_env.helper_functions import run_single_timestep_pf
import rl_charging_station.simulator.energy_calculations as energy_calculations


# TODO: make better sense of extracting actions (maybe move it to helper funtions etc)
def extract_actions(instance, action_array):
    # TODO check again if that makes sense!
    ev_actions = None
    ess_actions = None

    if instance.cs_action_start_idx is not None:
        ev_actions = action_array[
                     instance.cs_action_start_idx:instance.cs_action_start_idx + instance.n_cars_bus * instance.n_bus]

    if instance.ess_action_start_idx is not None:
        ess_actions = action_array[instance.ess_action_start_idx:instance.ess_action_start_idx + instance.n_ess]

    return ev_actions, ess_actions



def calculate_p_cs(cs_actions, n_cs, present_cars, timestep, soc_cs, cs_cap, cs_p_max):
    p_cs = np.zeros(n_cs)  # Initialize p_cs with zeros for each charging station
    penalties = np.zeros(n_cs)  # Initialize penalties with zeros

    for cs in range(n_cs):
        if present_cars[cs, timestep] == 1:  # Check if a car is present

            if cs_actions[cs] >= 0:  # Charging
                desired_p_cs = cs_actions[cs] * cs_p_max[cs]
                feasible_p_cs = calculate_max_charging_power_cs(cs, timestep, soc_cs, cs_cap[cs], cs_p_max[cs])
            else:  # Discharging
                desired_p_cs = cs_actions[cs] * cs_p_max[cs]
                feasible_p_cs = calculate_max_discharging_power_cs(cs, timestep, soc_cs, cs_cap[cs], cs_p_max[cs])

            p_cs[cs] = min(desired_p_cs, feasible_p_cs) if cs_actions[cs] >= 0 else max(desired_p_cs, feasible_p_cs)
            # TODO night make more sense to move the penalty calculation to the reward calculation section
            if (cs_actions[cs] >= 0 and desired_p_cs > feasible_p_cs) or (cs_actions[cs] < 0 and desired_p_cs < feasible_p_cs):
                penalties[cs] = (desired_p_cs - feasible_p_cs) ** 2  # Squared difference as penalty

        else:
            p_cs[cs] = 0  # No charging or discharging if no car is present

    return p_cs, penalties


def calculate_max_charging_power_cs(car, timestep, soc_cs, cs_cap, cs_p_max):
    remaining_capacity_kWh = (1 - soc_cs[car, timestep]) * cs_cap
    remaining_capacity_power_kw = remaining_capacity_kWh * 4  # Convert to kW for a 15-minute interval
    max_cs = min(cs_p_max, remaining_capacity_power_kw)
    return max_cs


def calculate_max_discharging_power_cs(car, timestep, soc_cs, cs_cap, cs_p_max):
    # Calculate the available capacity to be discharged in kWh
    available_capacity_kWh = soc_cs[car, timestep] * cs_cap
    # Convert the available capacity to an equivalent power value for the 15-minute timestep
    available_capacity_power_kw = available_capacity_kWh * 4  # Convert to kW for a 15-minute interval
    # The maximum feasible discharging power for this timestep is the lesser of the station's maximum output capability
    # or the power equivalent of discharging the available battery capacity in this timestep
    max_cs = min(cs_p_max, available_capacity_power_kw)
    # Ensure discharging does not result in negative SOC
    max_cs = max(-max_cs, -soc_cs[car, timestep] * cs_cap * 4)
    return max_cs


def calculate_p_ess(ess_actions, n_ess, timestep, soc_ess, ess_cap, ess_p_max):
    p_ess = np.zeros(n_ess)  # Initialize p_ess with zeros for each ESS unit
    penalties_ess = np.zeros(n_ess)  # Initialize penalties with zeros

    for ess in range(n_ess):
        if ess_actions[ess] >= 0:  # Charging
            desired_p_ess = ess_actions[ess] * ess_p_max[ess]
            feasible_p_ess = calculate_max_charging_power_ess(ess, timestep, soc_ess, ess_cap[ess], ess_p_max[ess])
        else:  # Discharging
            desired_p_ess = ess_actions[ess] * ess_p_max[ess]
            feasible_p_ess = calculate_max_discharging_power_ess(ess, timestep, soc_ess, ess_cap[ess], ess_p_max[ess])

        p_ess[ess] = min(desired_p_ess, feasible_p_ess) if ess_actions[ess] >= 0 else max(desired_p_ess, feasible_p_ess)

        if (ess_actions[ess] >= 0 and desired_p_ess > feasible_p_ess) or (ess_actions[ess] < 0 and desired_p_ess < feasible_p_ess):
            penalties_ess[ess] = (desired_p_ess - feasible_p_ess) ** 2  # Squared difference as penalty

    return p_ess, penalties_ess


def calculate_max_charging_power_ess(ess, timestep, soc_ess, ess_cap, ess_p_max):
    remaining_capacity_kWh = (1 - soc_ess[ess, timestep]) * ess_cap
    remaining_capacity_power_kw = remaining_capacity_kWh * 4  # Assuming 15-minute intervals
    max_ess_charging = min(ess_p_max, remaining_capacity_power_kw)
    return max_ess_charging


def calculate_max_discharging_power_ess(ess, timestep, soc_ess, ess_cap, ess_p_max):
    available_capacity_kWh = soc_ess[ess, timestep] * ess_cap
    available_capacity_power_kw = available_capacity_kWh * 4  # Assuming 15-minute intervals
    max_ess_discharging = min(ess_p_max, available_capacity_power_kw)
    max_ess_discharging = max(-max_ess_discharging, -soc_ess[ess, timestep] * ess_cap * 4)  # Ensure SOC does not go negative
    return max_ess_discharging



def update_soc_cs(n_cs, present_cars, timestep, soc_cs, p_charging, cs_cap):
    for cs in range(n_cs):
        if present_cars[cs, timestep] == 1:
            # TODO we do /4 because timestep now is 15 minutes not 60, so the p_cs is only valid for 15 minutes
            # Calculate the new SOC, ensuring the result is for a 15-minute timestep
            new_soc = soc_cs[cs, timestep] + p_charging[cs] / 4 / cs_cap[cs]

            new_soc = round(new_soc, 4)
            # Update SOC for the next timestep
            soc_cs[cs, timestep + 1] = new_soc

    return soc_cs


def update_soc_ess(n_ess, ess_active, timestep, soc_ess, p_ess, ess_cap):
    for ess in range(n_ess):
        if ess_active[ess, timestep] == 1:  # Assuming 'ess_active' indicates if the ESS is active or not
            # The division by 4 accounts for the conversion from power (kW) to energy (kWh) over a 15-minute interval
            new_soc = soc_ess[ess, timestep] + p_ess[ess] / 4 / ess_cap[ess]

            # Ensure the SOC stays within the 0 to 1 range and rounding to 4 decimal places for precision
            new_soc = round(max(0, min(new_soc, 1)), 4)

            # Update SOC for the next timestep
            soc_ess[ess, timestep + 1] = new_soc
        else:
            # If the ESS is not active, carry forward the SOC from the current timestep
            soc_ess[ess, timestep + 1] = soc_ess[ess, timestep]

    return soc_ess

def simulate_control(instance, actions):
    """
    Simulate the control of charging electric vehicles based on renewable energy sources,
    vehicle presence, and other metrics.

    Parameters:
    - instance (Object): An instance of a class containing attributes and methods related to the simulation
                         environment, such as the number of cars, buses, battery state of charge (BoC), energy
                         prices, etc.
    - actions (list or ndarray): A list or array containing the charging actions for each car. Positive values
                                 indicate charging, negative values indicate discharging.


    """
    # TODO: if network is simulated i need to have different cost functions, involving pandapower. there needs to be a
    #  distinction

    timestep = instance.timestep

    ev_actions, ess_actions = extract_actions(instance, actions)

    actions = {'ev_actions': ev_actions,
               'ess_actions': ess_actions}

    present_cars = instance.init_values_cs['present_cars']

    n_cs = instance.n_cs

    n_ess = instance.n_ess

    # TODO: Move everything below out of here, since its not really calculating the actions, more simulating their
    #  output you know?
    # TODO: introduce car class, and make boc part of each car itself and recalculating a method?
    if n_cs > 0:
        soc_cs = instance.soc_cs
        cs_cap = instance.cs_cap_kWh
        cs_p_max = instance.cs_p_max
        p_cs, penalties_os_p_cs = calculate_p_cs(ev_actions, n_cs, present_cars, timestep, soc_cs, cs_cap, cs_p_max)
        soc_cs = update_soc_cs(n_cs, present_cars, timestep, soc_cs, p_cs,
                               cs_cap)  # TODO: cs_cap instead of ev_capacity

    if n_ess > 0:
        soc_ess = instance.soc_ess
        ess_cap = instance.ess_cap_kWh
        ess_p_max = instance.ess_p_max

        p_ess, penalties_os_p_ess = calculate_p_ess(ess_actions, n_ess, timestep, soc_ess, ess_cap, ess_p_max)
        soc_ess = update_soc_ess(n_ess, timestep, soc_ess, p_ess, ess_cap)

    else:
        p_ess = 0  # TODO 0 is not so nice maybe better None?! --> more adjustments needed
        soc_ess = 0
        penalties_os_p_ess = 0

    net_result_dict = None
    if instance.simnet_flag:
        net_result_dict, net = run_single_timestep_pf(instance, time_step=timestep,
                                                      profile_data_dict=instance.profile_data_dict,
                                                      bus_dict=instance.bus_dict, loc_cs=instance.loc_cs,
                                                      loc_ess=instance.loc_ess, idx_dict=instance.idx_dict, cs_action_p=p_cs,
                                                      ess_action_p=p_ess)

        # TODO NEXT: not complete yet. i need a better logic for division of simnet or not and cost funtiuon for multiple is not done yet
        # also i wonder what adjustments need to be made for multi agent stuff!!
    else:
        # TODO implement the logic!
        # TODO: check if necessary here or in reward calculation
        pass
        # bus_energies = compute_bus_powers(instance, renewable_list, p_cs, p_ess, timestep)
        # res_avail = max([0, renewable[timestep] - instance.energy['consumed'][0, timestep]])
        # total_charging = sum(p_cs)
        # grid_final = max([total_charging - res_avail, 0])

    # TODO: pretty sure there is a logic mistake. check on res_available again!!

    bus_energies, network_energy = energy_calculations.compute_bus_powers(instance, p_cs, p_ess, timestep)
    total_grid_energy, total_res_available, total_net_energy = energy_calculations.compute_network_energy(instance, bus_energies)

    grid_final = total_net_energy
    res_avail = total_res_available

    if instance.debug_flag and timestep == 48:
        print(f"total_net_energy: {total_net_energy}")  # @ bus 0 we need to consider external grid for the powerbalance.
        print(f"network_energy: {network_energy}")  # valid value seems to be total_net_energy_- net_energy_bus

    action_results = {'timestep': timestep,
                      'actions': actions,
                      'grid_final': grid_final,
                      'res_avail': res_avail,
                      'soc_cs': soc_cs,
                      'soc_ess': soc_ess,
                      'p_cs': p_cs,
                      'penalties_os_p_cs': penalties_os_p_cs,
                      'penalties_os_p_ess': penalties_os_p_ess,
                      'p_ess': p_ess,
                      'net_result_dict': net_result_dict,
                      'bus_energy': bus_energies
                      }

    return action_results
