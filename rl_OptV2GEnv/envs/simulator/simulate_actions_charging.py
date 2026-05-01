import numpy as np
from network_env.helper_functions import run_single_timestep_pf
import rl_OptV2GEnv.envs.simulator.energy_calculations as energy_calculations


def extract_actions(instance, action_array):
    # TODO check again if that makes sense!
    cs_actions = None
    ess_actions = None

    if instance.cs_action_start_idx is not None:
        cs_actions = action_array[instance.cs_action_start_idx:instance.cs_action_start_idx + instance.n_cs]

    if instance.ess_action_start_idx is not None:
        ess_actions = action_array[instance.ess_action_start_idx:instance.ess_action_start_idx + instance.n_ess]

    return cs_actions, ess_actions


def simulate_actions(instance, actions):
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
    timestep = instance.timestep
    cs_actions, ess_actions = extract_actions(instance, actions)

    actions = {'cs_actions': cs_actions,
               'ess_actions': ess_actions}

    bus_dict = instance.bus_dict
    present_cars = instance.init_values_cs_ep['present_cars']
    n_cs = instance.n_cs
    n_ess = instance.n_ess

    ##############################
    # calculate action outcome CS
    ############################
    if n_cs > 0:
        soc_cs = instance.soc_cs
        p_cs_filled = []
        index = 0
        p_cs = np.zeros(n_cs)  # Initialize p_cs with zeros for each charging station
        penalties_cs = np.zeros(n_cs)  # Initialize penalties with zeros
        for bus_id, bus in bus_dict.items():
            if bus.has_cs:
                cs = bus.cs
                cs.action_t = cs_actions[index]
                cs.soc_cs_t = soc_cs[index, timestep]
                cs.soc_cs_next_t = soc_cs[index, timestep + 1]    # TODO only needed for non present cars, otherwise model breaks.. fix!
                cs.ev_present_t = present_cars[index, timestep]

                p_cs[index], soc_cs[index, timestep + 1], penalties_cs[index] = cs.calculate_p_soc_cs()
                p_cs_filled.append(p_cs[index])
                index += 1

            else:
                p_cs_filled.append(0)

    else:
        p_cs = 0
        soc_cs = 0
        penalties_cs = 0

    ##############################
    # calculate action outcome ESS
    ############################

    if n_ess > 0:
        soc_ess = instance.soc_ess
        p_ess_filled = []
        index = 0
        p_ess = np.zeros(n_ess)  # Initialize p_ess with zeros for each ESS unit
        penalties_ess = np.zeros(n_ess)  # Initialize penalties with zeros
        for bus_id, bus in bus_dict.items():
            if bus.has_ess:
                ess = bus.ess
                ess.soc_ess_t = soc_ess[index, timestep]
                ess.action_t = cs_actions[index]

                p_ess[index], soc_ess[index, timestep + 1], penalties_ess[index] = ess.calculate_p_soc_ess()
                p_ess_filled.append(p_ess[index])
                index += 1
            else:
                p_ess_filled.append(0)

    else:
        p_ess = 0
        soc_ess = 0
        penalties_ess = 0

    net_result_dict = None

    if instance.simnet_flag:
        net_result_dict, net = run_single_timestep_pf(instance, time_step=timestep,
                                                      bus_dict=instance.bus_dict, loc_cs=instance.loc_cs,
                                                      loc_ess=instance.loc_ess, idx_dict=instance.idx_dict, cs_action_p=p_cs,
                                                      ess_action_p=p_ess)




    # TODO: pretty sure there is a logic mistake. check on res_available again!!

    bus_energies = energy_calculations.compute_bus_powers(instance, timestep)
    total_grid_energy, total_pv_available, total_net_energy, network_energy_timestep = energy_calculations.compute_network_power(instance, bus_energies)

    grid_final = total_net_energy
    pv_available = total_pv_available

    if instance.debug_flag and timestep == 48:
        print(f"total_net_energy: {total_net_energy}")  # @ bus 0 we need to consider external grid for the powerbalance.
        print(f"network_energy: {network_energy_timestep}")  # valid value seems to be total_net_energy_- net_energy_bus

    action_results = {'timestep': timestep,
                      'actions': actions,
                      'grid_final': grid_final,
                      'pv_available': pv_available,
                      'soc_cs': soc_cs,
                      'soc_ess': soc_ess,
                      'p_cs': p_cs,
                      'penalties_cs': penalties_cs,
                      'penalties_ess': penalties_ess,
                      'p_ess': p_ess,
                      'net_result_dict': net_result_dict,
                      'bus_energy': bus_energies,
                      'p_cs_filled': p_cs_filled,
                      'p_ess_filled': p_ess_filled,
                      }

    return action_results
