from datetime import datetime
import time


# TODO:

# reward weights:
#       i'd also like to balance o
# wm: main weights for:
#       - lines
#       - energy price
#       - fully loaded cars
#       - bus / network autarky
# ws: sub weights for:
#       - lines


def calculate_cost_per_bus(instance, energy_price_norm_t, energy_price_real_t, action_results, energy_price_range):
    """
    Calculate the cost at every bus.

    Parameters:
    - instance: The class instance with all necessary attributes and methods.

    Returns:
    - bus_costs: A dictionary with bus identifiers as keys and their respective costs as values.
    """
    # for normalized:
    # buying_adjustment = (0.03 * energy_price_norm_t + 0.1671 / energy_price_range)  # 3% increase and +0.1671 €/kWh adjustment
    # selling_adjustment = 0.05 / energy_price_range  # Assuming energy_price_range is max_price - min_price from original prices
    # for real prices:
    buying_price = 1.03 * energy_price_norm_t + 0.1671 * energy_price_range
    selling_price = energy_price_norm_t - 0.05 * energy_price_range

    for bus_id, bus in instance.bus_dict.items():
        # self_consumption = calculate_self_consumption(instance, bus_id, bus)

        # TODO NEXT: check sign of energy prices: feed-in and buying energy must be considered oppositely
        if bus.p_net_t >= 0:
            # adjusted_buying_price = energy_price_norm_t + buying_adjustment
            # grid_cost_bus = bus_energy[bus_id]['from_grid'] * adjusted_buying_price / 4
            bus.grid_cost_t = bus.p_from_grid_t * buying_price / 4
            action_results['bus_energy'][bus_id]['energy_cost'] = bus.grid_cost_t
            action_results['bus_energy'][bus_id]['feed-in_gain'] = 0
            bus.grid_revenue_t = 0

        else:
            # adjusted_selling_price = energy_price_norm_t - selling_adjustment
            bus.grid_revenue_t = bus.p_to_grid_t * selling_price / 4
            action_results['bus_energy'][bus_id]['energy_cost'] = 0
            action_results['bus_energy'][bus_id]['feed-in_gain'] = bus.grid_revenue_t
            bus.grid_cost_t = 0

    return action_results


def calculate_cost_netsim(net_result_dict, debug_flag):
    """
    placeholder for costs -> reward of network simulation
    :return:
    """

    ws_l100 = 5
    ws_l80 = 1
    ws_l50 = 0.2
    ws_l20 = 0


    cost_lines_total = 0

    for line_id in range(len(net_result_dict['res_line'])):
        loading_percent = net_result_dict['res_line'][line_id]
        if loading_percent / 100 > 1:
            if debug_flag:
                print(f"exceeded limit: {loading_percent}% on line: {line_id}")
            cost_line_loading = ws_l100

        elif loading_percent / 100 > 0.8:
            cost_line_loading = ws_l80

        elif loading_percent / 100 > 0.5:
            cost_line_loading = ws_l50

        elif loading_percent / 100 < 20:
            cost_line_loading = ws_l20

        cost_lines_total += cost_line_loading

    return cost_lines_total


def calculate_cost_ext_grid(instance, res_ext_grid_p_mw, energy_price_norm_t, energy_price_range):
    bus_dict = instance.bus_dict
    sum_bus_grid_cost = sum(bus.grid_cost_t for bus in bus_dict.values())
    sum_bus_revenue = sum(bus.grid_revenue_t for bus in bus_dict.values())

    # TODO maybe only penalty if cost is preventable, like if car to discharge or ess is avalable.
    # in interactive plot implement grid power, maybe easier to evaluate then.
    # Calculate the relative adjustments based on the energy price range
    buying_adjustment = (0.03 * energy_price_norm_t + 0.1671 / energy_price_range)  # 3% increase and +0.1671 €/kWh adjustment
    selling_adjustment = 0.05 / energy_price_range  # Assuming energy_price_range is max_price - min_price from original prices

    if res_ext_grid_p_mw >= 0:
        # Apply the buying adjustment directly
        adjusted_buying_price = energy_price_norm_t + buying_adjustment
        buying_cost_ext_grid = res_ext_grid_p_mw * 1000 * adjusted_buying_price / 4  # Assuming 15-minute intervals
        selling_cost_ext_grid = 0

    else:
        # Apply the selling adjustment directly
        adjusted_selling_price = energy_price_norm_t - selling_adjustment
        selling_cost_ext_grid = res_ext_grid_p_mw * 1000 * adjusted_selling_price / 4  # Assuming 15-minute intervals
        buying_cost_ext_grid = 0

    return sum_bus_grid_cost, sum_bus_revenue, selling_cost_ext_grid, buying_cost_ext_grid


def penalty_remaining_pv(instance):
    penalty_sum_remaining_pv = 0
    timestep = instance.timestep
    for bus_id, bus in instance.bus_dict.items():
        # Initialize conditions as False
        cs_condition, ess_condition = False, False

        if bus.has_pv:
            if bus.has_cs:
                soc_cs = bus.cs.soc_cs_t
                cs_condition = soc_cs != -1 and soc_cs <= 0.92 and bus.cs.cs_p_max - 0.2 > bus.cs.p_cs_t  # SoC for CS is neither -1 nor full

            if bus.has_ess:
                # Find the index for this bus_id in the soc_ess array using loc_ess
                soc_ess = bus.ess.soc_ess_t  # SoC at ESS
                ess_condition = soc_ess <= 0.92 and bus.ess.ess_p_max - 0.2 > bus.ess.p_ess_t_cs_t  # SoC for ESS is not full

            # Apply penalty if either CS or ESS conditions are met
            if cs_condition or ess_condition:
                pv_bus = bus.pv.pv_prod_episode[timestep]
                pv_available = bus.pv_available_t

                if pv_bus > 0:
                    unused_fraction = pv_available / pv_bus  # Calculate the fraction of unused PV production
                    penalty_sum_remaining_pv += unused_fraction  # Add penalty based on unused PV
                else:
                    penalty_sum_remaining_pv += 0  # No penalty if there's no PV production
            else:
                penalty_sum_remaining_pv += 0  # No penalty if neither CS nor ESS conditions are met
        else:
            penalty_sum_remaining_pv += 0

    return penalty_sum_remaining_pv


def penalty_soc_departure_cs(instance, timestep, leave, soc_cs):
    """
    Calculate the total penalty for electric vehicles (EVs) based on their state of charge (SoC) after a given timestep.
    This function computes the cost for each EV that is indicated to leave after the current timestep. The cost is calculated as the squared difference between the ideal SoC (1, or fully charged) and the actual SoC after the timestep, multiplied by 2. This approach penalizes deviations from the ideal SoC.
    :param instance:
    :param timestep:
    :param leave: was calculated in the last observation. is telling if cs leaves AFTER? - "in the end" of - this timestep.
    :param soc_cs: soc_cs[cs, timestep + 1] describes the soc of the cs AFTER? this timestep.
    :return:
    """

    # TODO: does it really make sense just to sum up all of the costs? would it be helpful to define a different metric?
    # like e.g. within the reward function some rating. excess of energy -5, perfect use of renewables = +100 etc.
    last_timestep = instance.last_timestep

    sum_penalty_cs = 0
    # Iterate through each cs in leave array
    for cs in leave:
        # Skip penalty calculation if this is the last timestep
        if timestep == last_timestep:
            break
        # Calculate penalty for cars not fully charged
        penalty_cs = ((1 - soc_cs[cs, timestep + 1]) * 2) ** 2
        sum_penalty_cs += penalty_cs

    return sum_penalty_cs


def penalty_action_empty_cs(instance, action_results):
    penalty_action = 0  # Initialize the penalty counter
    timestep = instance.timestep
    cs_actions = action_results['actions']['cs_actions']

    for cs, ev_action in enumerate(cs_actions):
        # Check if the SoC for the current car at the current timestep is -1
        if instance.init_values_cs_ep['present_cars'][cs][timestep] == 0:
            # If the action for this car is not zero, add 5 to the penalty
            if ev_action != 0:
                penalty_action += 5

    return penalty_action


def calculate_reward(instance, action_results):
    timestep = instance.timestep
    rew_penal_dict = {}  # dict for partial rewards and penalties, also for logging in tensorboard

    objective_weights = instance.objective_weights
    wn_reward = 0  # weighted and normalized/scaled final rewards. partial costs / penalties will be added continuously

    energy_price_norm_t = instance.da_price_ep_norm[timestep]  # normalized energy_price_norm_t of timestep
    energy_price_real_t = instance.da_price_ep[timestep]
    energy_price_range = instance.da_price_range

    # TODO: think of the case where this is even needed. might be sufficient to take the total energy price of the overall ext_grid
    #########################
    # CALCULATE Partial Costs:
    # calculate cost per bus (energy that bus buys @ price of current timestep or sells @ price of current timestep - offset

    ##########
    # Scaling
    #########
    scale_lines_penalty = 1
    scale_ev = 1
    # norm_ext_grid = 1 / instance.n_bus
    # norm_pv = 1 / 50
    scale_os_p_cs = 1 / 1000

    #####################################
    # IMPORTANT Penalties / Rewards considering Energy-Price
    ###################################
    # TODO: distinct reward for optimising total ext_grid consumption of the network and optimising energy from grid per bus??
    #  sum_buses_grid_cost != cost_ext_grid
    # TODO: sum_buses_grid_cost, cost_ext_grid might be redundant
    if instance.simnet_flag:
        res_ext_grid_p_mw = instance.net.res_ext_grid['p_mw'][0]  # TODO: check if i even need this
    else:
        res_ext_grid_p_mw = action_results['grid_final'] / 1000

    action_results = calculate_cost_per_bus(instance=instance, energy_price_norm_t=energy_price_norm_t,
                                            energy_price_real_t=energy_price_real_t,
                                            action_results=action_results,
                                            energy_price_range=energy_price_range)

    sum_buses_grid_cost, sum_buses_revenue, selling_cost_ext_grid, buying_cost_ext_grid = calculate_cost_ext_grid(instance,
                                                                                                                  res_ext_grid_p_mw,
                                                                                                                  energy_price_norm_t,
                                                                                                                  energy_price_range)
    ##############################################################
    # penalty for buying energy from the grid NOTE: should be weighted equal to selling reward for realistic use case
    if objective_weights['wm_ep_b'] > 0:
        wn_buying_penalty_ext_grid = objective_weights['wm_ep_b'] * sum_buses_grid_cost
        # TODO: dividing through for scaling: / instance.n_bus  # / instance.n_bus seemed to much for more then two busses
        rew_penal_dict['wn_buying_ext_grid'] = -wn_buying_penalty_ext_grid
        wn_reward += wn_buying_penalty_ext_grid
    ##############################################################
    # reward for selling energy to the grid NOTE: should be weighted equal to buying penalty for realistic use case
    if objective_weights['wm_ep_s'] > 0:
        wn_selling_reward_ext_grid = objective_weights['wm_ep_s'] * sum_buses_revenue  # / instance.n_bus  # * norm_ext_grid
        rew_penal_dict['wn_selling_ext_grid'] = -wn_selling_reward_ext_grid
        wn_reward += wn_selling_reward_ext_grid

    # EXPERIMENTAL not really used, because looking at every bus individually, hard to optimize!
    if objective_weights['wm_c_sum_be'] > 0:
        rew_penal_dict['cost_sum_bus_energy'] = -sum_buses_grid_cost

    ##############################################################
    # IMPORTANT penalty for not charging cars sufficiently on departure
    if objective_weights['wm_cs'] > 0:
        penalty_sum_soc_cs = penalty_soc_departure_cs(instance=instance, timestep=timestep, leave=instance.leave, soc_cs=instance.soc_cs)
        wn_penalty_sum_soc_cs = objective_weights['wm_cs'] * penalty_sum_soc_cs * scale_ev
        rew_penal_dict['penalty_ev'] = - wn_penalty_sum_soc_cs
        wn_reward += wn_penalty_sum_soc_cs

    ##############################################################
    # penalty for wasting available pv power, makes sense if energy is not sold -> non-v2g scenario or for having high self-sufficiency
    if objective_weights['wm_rp'] > 0:
        penalty_sum_remaining_pv = penalty_remaining_pv(instance)
        wn_penalty_sum_remaining_pv = objective_weights['wm_rp'] * penalty_sum_remaining_pv  # / instance.n_pv  # norm_pv
        rew_penal_dict['penalty_sum_remaining_pv'] = -wn_penalty_sum_remaining_pv
        wn_reward += wn_penalty_sum_remaining_pv

    ##############################################################
    # EXPERIMENTAL penalty for overshooting feasible charging through action, even though station empty
    if objective_weights['wm_os_c'] > 0:
        sum_penalties_os_p_cs = sum(action_results['penalties_cs'])  # TODO: remove!

        sum_penalties_os_p_cs = sum(bus.cs.penalty_os_t for bus in instance.bus_dict.values() if bus.has_cs)
        # TODO check if its the same!
        if sum_penalties_os_p_cs != sum(action_results['penalties_cs']):
            print('unexpected!!')

        wn_penalties_os_p_cs = objective_weights['wm_os_c'] * scale_os_p_cs * sum_penalties_os_p_cs
        rew_penal_dict['sum_penalties_os_p_cs'] = -sum_penalties_os_p_cs
        wn_reward += wn_penalties_os_p_cs

    ###############################################################
    # EXPERIMENTAL penalty for charging, even though station empty
    if objective_weights['wm_empty_station'] > 0:
        penalty_sum_action_empty_cs = penalty_action_empty_cs(instance, action_results)
        wn_penalty_sum_action_empty_cs = objective_weights['wm_empty_station'] * penalty_sum_action_empty_cs / instance.n_cs
        rew_penal_dict['penalty_sum_action_empty_cs'] = -penalty_sum_action_empty_cs
        wn_reward += wn_penalty_sum_action_empty_cs

    ########################################
    # penalty for lines, when simnet active
    if instance.simnet_flag:
        cost_lines_total = calculate_cost_netsim(action_results['net_result_dict'], instance.debug_flag)
        wn_penalty_lines_total = 0

        if objective_weights['wm_l'] > 0:
            wn_penalty_lines_total = objective_weights['wm_l'] * cost_lines_total * scale_lines_penalty
            rew_penal_dict['penalty_lines_total'] = -wn_penalty_lines_total

        wn_reward += wn_penalty_lines_total

    #####################################
    # Sum of Total Rewards and Penalties
    ###################################
    rew_penal_dict['reward'] = -wn_reward

    # if instance.simnet_flag:
    #     pass
    # # TODO: I'm really not sure if this works properly!!
    # else:
    #     pass
    #     action_results = calculate_cost_per_bus(instance=instance, energy_price_norm_t=energy_price_norm_t,
    #                                             action_results=action_results)
    #
    #     total_energy_cost, cost_ev = calculate_cost_ev(timestep=timestep, simnet_flag=instance.simnet_flag,
    #                                                    energy_price_norm_t=energy_price_norm_t,
    #                                                    grid_final=action_results['grid_final'],
    #                                                    leave=instance.leave, soc_cs=instance.soc_cs,
    #                                                    action_results=action_results,
    #                                                    net_result_dict=action_results['net_result_dict'])
    #
    #     cost = total_energy_cost

    return wn_reward, rew_penal_dict, buying_cost_ext_grid, selling_cost_ext_grid
