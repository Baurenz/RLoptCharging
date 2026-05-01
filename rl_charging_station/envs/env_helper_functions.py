import numpy as np
from gymnasium import spaces


def create_action_space(instance):
    total_action_size = 0
    cs_action_start_idx = None
    ess_action_start_idx = None

    # Initialize low and high bounds for the action space
    low = []
    high = []

    # Calculate the size and start indices for each action space
    if instance.n_cs > 0:
        cs_action_start_idx = total_action_size
        for bus in instance.bus_dict.values():
            if bus.has_cs:
                if bus.has_v2g:
                    # For V2G capable buses, actions can range from -1 to 1
                    low.append(-1)
                else:
                    # For non-V2G buses, actions can range from 0 to 1
                    low.append(0)
                high.append(1)  # Upper bound is always 1 for CS
        instance.total_action_size += instance.n_cs

    if instance.n_ess > 0:
        ess_action_start_idx = total_action_size
        # ESS actions range from -1 to 1, so append -1 and 1 to low and high bounds for each ESS
        low.extend([-1] * instance.n_ess)
        high.extend([1] * instance.n_ess)
        total_action_size += instance.n_ess

    # Convert low and high lists to numpy arrays
    low_array = np.array(low, dtype=np.float32)
    high_array = np.array(high, dtype=np.float32)

    # Create a single, flat action space with custom bounds
    action_space = spaces.Box(low=low_array, high=high_array, dtype=np.float32)

    return action_space, total_action_size, cs_action_start_idx, ess_action_start_idx


def save_decision_results(instance):
    cs_soc_episode = instance.init_values_cs['soc_cs']
    departure_t = instance.init_values_cs['departure_t']

    # Initialize the list to hold the SoC values for each car at departure times
    cs_soc_leave = []

    # Iterate over each car and its departure times
    for car_index, departures in enumerate(departure_t):
        # Extract the SoC values for the current car at the specified departure times
        socs_at_departure = cs_soc_episode[car_index, departures]
        # Convert the NumPy array to a list and append it to the cs_soc_leave list
        cs_soc_leave.append(socs_at_departure.tolist())

    decision_result_dict = {'cs_soc_leave': cs_soc_leave}

    return decision_result_dict


def create_observation_space(instance):
    obs_timestep = 1
    obs_price_n = 1 + instance.n_pred_price
    obs_pv_n = instance.solar_flag * instance.n_pv * (1 + instance.n_pred_pv)
    obs_load_n = instance.n_load
    obs_cs_n = 2 * instance.n_cs
    obs_ess_n = instance.n_ess

    dict_obs_n = {
        'obs_price_n': obs_price_n,
        'obs_pv_n': obs_pv_n,
        'obs_load_n': obs_load_n,
        'obs_cs_n': obs_cs_n,
        'obs_ess_n': obs_ess_n,
        'obs_cs_ess_states_n': obs_cs_n + obs_ess_n
    }

    observation_n = obs_timestep + obs_cs_n + obs_ess_n + obs_pv_n + obs_load_n + obs_price_n # self.simnet_flag * 1 +

    low = np.array(-np.ones(observation_n), dtype=np.float32)
    high = np.array(np.ones(observation_n), dtype=np.float32)

    # TODO: divide ranges of observation space! for price [-1,1] for other values [0,1]
    observation_space = spaces.Box(
        low=low,
        high=high,
        dtype=np.float64  # TODO: think of Datatype: float32 can lead to less effort how can i change this?!
    )

    return observation_space, dict_obs_n
