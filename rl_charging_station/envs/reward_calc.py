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


wm_ev = 0  # weight to reach ev charging goals
wm_ep_b = 150  # weight for energy price buying
wm_ep_s = 0
wm_ss = 300  # weight for self-sufficiency
wm_os_c = 0

wm_l = 1
ws_l100 = 5
ws_l80 = 1
ws_l50 = 0.2
ws_l20 = 0.002
ws_l0 = 0


def calculate_cost_per_bus(instance, energy_price, action_results):
    """
    Calculate the cost at every bus.

    Parameters:
    - instance: The class instance with all necessary attributes and methods.

    Returns:
    - bus_costs: A dictionary with bus identifiers as keys and their respective costs as values.
    """
    bus_energy = action_results['bus_energy']
    # Iterate over each bus in the bus_dict
    for bus_id, bus_data in instance.bus_dict.items():
        # self_consumption = calculate_self_consumption(instance, bus_id, bus_data)

        # TODO NEXT: check sign of energy prices: feed-in and buying energy must be considered oppositely
        if bus_energy[bus_id]['net_power_bus'] <= 0:
            feedin_gain = bus_energy[bus_id]['to_grid'] * (energy_price - 0.15) / 4  # TODO: feed-in tariff 'more expensive' then buying!!
            #  now only for 15 minutes
            action_results['bus_energy'][bus_id]['energy_cost'] = 0
            action_results['bus_energy'][bus_id]['feed-in_gain'] = feedin_gain
        else:
            grid_cost_bus = bus_energy[bus_id]['from_grid'] * energy_price / 4  # # TODO: now only for 15 minutes

            action_results['bus_energy'][bus_id]['energy_cost'] = grid_cost_bus
            action_results['bus_energy'][bus_id]['feed-in_gain'] = 0

    # TODO: check if i need both in reward function maybe its sufficient to use the costs?

    return action_results


def calculate_cost_netsim(net_result_dict, ws_l100, ws_l80, ws_l50, ws_l20, ws_l0, debug_flag):
    """
    placeholder for costs -> reward of network simulation
    TODO: implement something meaningful.
    :return:
    """
    cost_lines_total = 0

    for line_id in range(len(net_result_dict['res_line']['loading_percent'])):
        loading_percent = net_result_dict['res_line']['loading_percent'][line_id]
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

        else:
            cost_line_loading = loading_percent / 1000

        cost_lines_total += cost_line_loading

    return cost_lines_total


def calculate_cost_ext_grid(res_ext_grid_p_mw, action_results, energy_price):
    sum_bus_energy_cost = sum(bus_data['energy_cost'] for bus_data in action_results['bus_energy'].values())
    # cost_ext_grid = action_results['grid_final'] * energy_price

    # TODO: check sign for power 'import' & 'export' does it make sense??
    if res_ext_grid_p_mw >= 0:
        buying_cost_ext_grid = res_ext_grid_p_mw * 1000 * energy_price / 4  # TODO: now only for 15 minutes
        selling_cost_ext_grid = 0

    else:
        selling_cost_ext_grid = res_ext_grid_p_mw * 1000 * (energy_price - 0.15) / 4  # TODO: now only for 15 minutes
        buying_cost_ext_grid = 0  # TODO: only for testing, if the car charges completely only when sun is there!

    return sum_bus_energy_cost, selling_cost_ext_grid, buying_cost_ext_grid


def penalty_remaining_pv(instance, action_results):
    penalty_sum_remaining_pv = 0

    timestep = action_results['timestep']

    for bus_id in action_results['bus_energy']:
        # Retrieve the state of charge for the given bus_id and timestep
        soc = action_results['soc_cs'][bus_id, timestep]

        # Check if SOC is neither -1 (car not present) nor full (<= 0.945)
        if soc != -1 and soc <= 0.945:
            pv_bus = instance.bus_dict[bus_id].pv_prod_episode[timestep]
            res_available = action_results['bus_energy'][bus_id]['res_available']

            if pv_bus > 0:
                # Calculate the fraction of unused PV production
                unused_fraction = res_available / pv_bus

                # Optionally the penalty to accentuate the effect of larger fractions
                # penalty = unused_fraction ** 2  # Square the fraction to make the penalty more severe for higher fractions

                penalty_sum_remaining_pv += unused_fraction
            else:
                # Handle cases where there is no PV production
                penalty_sum_remaining_pv += 0  # No penalty if there's no PV production

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

    last_timestep = instance.last_timestep

    # TODO: does it really make sense just to sum up all of the costs? would it be helpful to define a different metric?
    # like e.g. within the reward function some rating. excess of energy -5, perfect use of renewables = +100 etc.

    sum_penalty_cs = 0

    # Iterate through each cs in leave array
    for cs in leave:
        # Skip penalty calculation if this is the last timestep
        if timestep == last_timestep:
            continue
        # Calculate penalty for cars not fully charged
        penalty_cs = ((1 - soc_cs[cs, timestep + 1]) * 2) ** 2
        sum_penalty_cs += penalty_cs

    return sum_penalty_cs


def calculate_reward(instance, action_results):
    timestep = instance.timestep

    # TODO: do I want to keep enegery price normalized?
    energy_price = instance.day_ahead_price_episode_norm[timestep]
    if instance.simnet_flag:
        res_ext_grid_p_mw = instance.net.res_ext_grid['p_mw'][0]
    else:
        res_ext_grid_p_mw = action_results['grid_final'] / 1000

    # TODO: think of the case where this is even needed. might be sufficient to take the total energy price of the overall ext_grid
    #########################
    # CALCULATE Partial Costs:
    # calculate cost per bus (energy that bus buys @ price of current timestep or sells @ price of current timestep - offset
    action_results = calculate_cost_per_bus(instance=instance, energy_price=energy_price, action_results=action_results)

    sum_penalties_os_p_cs = sum(action_results['penalties_os_p_cs'])

    # weights to be found on top of this file

    if instance.simnet_flag:
        cost_lines_total = calculate_cost_netsim(action_results['net_result_dict'], ws_l100, ws_l80, ws_l50, ws_l20, ws_l0,
                                                 instance.debug_flag)
    else:
        cost_lines_total = 0

    cost_sum_bus_energy, selling_cost_ext_grid, buying_cost_ext_grid = calculate_cost_ext_grid(res_ext_grid_p_mw, action_results,
                                                                                               energy_price)
    # TODO: cost_sum_bus_energy, cost_ext_grid might be redundant

    penalty_sum_soc_cs = penalty_soc_departure_cs(instance=instance, timestep=timestep, leave=instance.leave, soc_cs=instance.soc_cs)
    penalty_sum_remaining_pv = penalty_remaining_pv(instance, action_results)

    # TODO: distinct reward function for optimising total ext_grid consumption of the network and optimising energy from grid per bus??
    #  cost_sum_bus_energy != cost_ext_grid

    #############
    ### NORMALIZATION
    ############
    # TODO: more sophisticated normalization, for now more approximated!!
    norm_lines = 1 / 10
    norm_ev = 1
    norm_ext_grid = 1 / 2
    norm_pv = 1 / 50
    norm_os_p_cs = 1 / 1000

    # weighted and 'normalized' partial costs --> wn
    wn_cost_lines_total = wm_l * cost_lines_total * norm_lines
    wn_cost_ev_total = wm_ev * penalty_sum_soc_cs * norm_ev
    wn_buying_cost_ext_grid = wm_ep_b * buying_cost_ext_grid * norm_ext_grid
    wn_selling_cost_ext_grid = wm_ep_s * selling_cost_ext_grid * norm_ext_grid

    wn_cost_sum_remaining_pv = wm_ss * penalty_sum_remaining_pv * norm_pv
    wn_penalties_os_p_cs = wm_os_c * norm_os_p_cs * sum_penalties_os_p_cs

    wn_cost = wn_cost_lines_total + wn_cost_ev_total + wn_buying_cost_ext_grid + wn_selling_cost_ext_grid + wn_cost_sum_remaining_pv + \
              wn_penalties_os_p_cs

    if penalty_sum_soc_cs > 0 and instance.debug_flag:
        print(f"---partial costs @ timestep: {timestep}:---------------------")
        print(f"cost_lines_total | weighted {wm_l} =", cost_lines_total)
        print(f"penalty_sum_soc_cs | weighted {wm_ev}=", penalty_sum_soc_cs)
        print(f"cost_ext_grid | weighted {wm_ep_b}=", buying_cost_ext_grid)
        print(f"penalty_sum_remaining_pv | weighted {wm_ss}=", penalty_sum_remaining_pv)
        print("------------------------------------------")
        print(f"---normalized and weighted partial costs @ timestep: {timestep}:---")
        print(f"cost_lines_total | normalized & weighted {wm_l} =", cost_lines_total * norm_lines * wm_l)
        print(f"penalty_sum_soc_cs | normalized & weighted {wm_ev}=", penalty_sum_soc_cs * norm_ev * wm_ev)
        print(f"cost_ext_grid | normalized & weighted {wm_ep_b}=", buying_cost_ext_grid * norm_ext_grid * wm_ep_b)
        print(f"penalty_sum_remaining_pv | normalized & weighted {wm_ss}=", penalty_sum_remaining_pv * norm_pv * wm_ss)
        print("------------------------------------------------------------")

    # if instance.simnet_flag:
    #     pass
    # # TODO: I'm really not sure if this works properly!!
    # else:
    #     pass
    #     action_results = calculate_cost_per_bus(instance=instance, energy_price=energy_price,
    #                                             action_results=action_results)
    #
    #     total_energy_cost, cost_ev = calculate_cost_ev(timestep=timestep, simnet_flag=instance.simnet_flag,
    #                                                    energy_price=energy_price,
    #                                                    grid_final=action_results['grid_final'],
    #                                                    leave=instance.leave, soc_cs=instance.soc_cs,
    #                                                    action_results=action_results,
    #                                                    net_result_dict=action_results['net_result_dict'])
    #
    #     cost = total_energy_cost

    return wn_cost, wn_cost_lines_total, wn_cost_ev_total, wn_buying_cost_ext_grid, wn_selling_cost_ext_grid, wn_cost_sum_remaining_pv, \
        cost_sum_bus_energy, buying_cost_ext_grid, selling_cost_ext_grid, wn_penalties_os_p_cs
