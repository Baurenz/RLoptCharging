import json
import os

import numpy as np
from gymnasium import spaces


def ndarray_to_list(data):
    if isinstance(data, dict):
        return {k: ndarray_to_list(v) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


class CsEnvHelper:

    def create_observation_space(self):
        obs_timestep = 1
        # obs_day_of_year_n = 2
        obs_price_n = 1 + self.n_pred_price
        obs_pv_n = self.n_pv * (1 + self.n_pred_pv)
        obs_load_n = self.n_load
        obs_cs_n = 3 * self.n_cs
        obs_ess_n = self.n_ess

        dict_obs_n = {
            'obs_price_n': obs_price_n,
            # 'obs_day_of_year': obs_day_of_year_n,
            'obs_pv_n': obs_pv_n,
            'obs_load_n': obs_load_n,
            'obs_cs_n': obs_cs_n,
            'obs_ess_n': obs_ess_n,
            'obs_cs_ess_states_n': obs_cs_n + obs_ess_n
        }
        # + obs_day_of_year_n
        observation_n = obs_timestep + obs_cs_n + obs_ess_n + obs_pv_n + obs_load_n + obs_price_n

        # Initialize low and high arrays with default range [0, 1]
        low = np.zeros(observation_n, dtype=np.float32)
        # low = -np.ones(observation_n, dtype=np.float32)

        high = np.ones(observation_n, dtype=np.float32)

        # Adjust the range for obs_timestep to [0, 1], which is already the default

        # Adjust the range for obs_cs_n to [-1, 1]
        # cs_start = obs_timestep
        # cs_end = cs_start + obs_cs_n
        # low[cs_start:cs_end] = 0
        # high[cs_start:cs_end] = 1

        observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        return observation_space, dict_obs_n

    def create_action_space(self):
        total_action_size = 0
        cs_action_start_idx = None
        ess_action_start_idx = None

        # Initialize low and high bounds for the action space
        low = []
        high = []

        # Calculate the size and start indices for each action space
        # if self.n_cs > 0:
        #     cs_action_start_idx = total_action_size
        #     for bus in self.bus_dict.values():
        #         if bus.has_cs:
        #             if bus.cs.v2g:
        #                 # For V2G capable buses, actions can range from -1 to 1
        #                 low.append(-1)
        #             else:
        #                 # For non-V2G buses, actions can range from 0 to 1
        #                 # turned out to be better with symmetrical action space and adjusting actions
        #                 low.append(-1)
        #             high.append(1)  # Upper bound is always 1 for CS
        #     total_action_size += self.n_cs

        # Calculate the size and start indices for each action space
        if self.n_cs > 0:
            cs_action_start_idx = total_action_size
            low.extend([-1] * self.n_cs)
            high.extend([1] * self.n_cs)
            total_action_size += self.n_cs

        if self.n_ess > 0:
            ess_action_start_idx = total_action_size
            # ESS actions range from -1 to 1, so append -1 and 1 to low and high bounds for each ESS
            low.extend([-1] * self.n_ess)
            high.extend([1] * self.n_ess)
            total_action_size += self.n_ess

        # Convert low and high lists to numpy arrays
        low_array = np.array(low, dtype=np.float32)
        high_array = np.array(high, dtype=np.float32)

        # Create a single, flat action space with custom bounds
        action_space = spaces.Box(low=low_array, high=high_array, dtype=np.float32)

        return action_space, cs_action_start_idx, ess_action_start_idx

    def save_decision_results(self):
        # TODO: actually sufficient to do this while evaluation. in here might be too costly
        cs_soc_episode = self.init_values_cs_ep['soc_cs']
        departure_t = self.init_values_cs_ep['departure_t']

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

    def store_evol(self, action_results, reward, buying_cost_ext_grid, selling_cost_ext_grid):
        # TODO: add evolutdion for all costs to see how they advance independently?
        # TODO: also add a list for history of charging power of ess

        p_cs = action_results['p_cs']
        p_ess = action_results['p_ess']

        if self.simnet_flag:
            self.ext_grid_evol.append(action_results['net_result_dict']['res_ext_grid']['p_mw'][0] * 1000)
            # TODO more consistency in kw / MW
            self.line_load_evol.append(action_results['net_result_dict']['res_line'].copy())

        else:
            self.ext_grid_evol.append(action_results['grid_final'])

        if self.n_ess > 0:
            self.ess_power_evol.append(p_ess.tolist())  # TODO not too neat!!
            self.p_cs_filledevol.append(action_results['p_cs_filled'])
        if self.n_cs > 0:
            self.cs_power_evol.append(p_cs.tolist())
            self.pv_wasted_evol.append(action_results['pv_available'])
            self.p_ess_filledevol.append(action_results['p_ess_filled'])

        if self.objective_weights['wm_cs'] > 0:
            self.penalty_ev_evol.append(self.rew_penal_dict['penalty_ev'])

        if self.objective_weights['wm_l'] > 0:
            self.penalty_lines_evol.append(self.rew_penal_dict['penalty_lines_total'])

        self.reward_evol.append(reward)
        self.bus_energies_evol.append(action_results['bus_energy'])
        self.cost_ext_grid_evol.append(buying_cost_ext_grid + selling_cost_ext_grid)

    def save_every_x_init_result(self, decision_result_dict):
        # TODO: lets also save relevant data per bus and more general maybe?!
        #  - line loading

        renewable_production = {i: self.bus_dict[i].pv.pv_prod_episode for i in self.bus_dict if self.bus_dict[i].has_pv}
        load = {i: self.bus_dict[i].load.load_episode for i in self.bus_dict if self.bus_dict[i].has_load}

        ##########################
        # Save the initial_values
        results = {'soc_cs': self.soc_cs,
                   'power_cs': self.cs_power_evol,
                   'cs_soc_leave': decision_result_dict['cs_soc_leave'],
                   'soc_ess': self.soc_ess,
                   'power_ess': self.ess_power_evol,
                   'renewable': renewable_production,
                   'load': load,
                   'res_wasted': self.pv_wasted_evol,
                   'bus_energies': self.bus_energies_evol,
                   'grid_final': self.ext_grid_evol,
                   'grid_cost': self.cost_ext_grid_evol,
                   'day_ahead_date': f'{self.date_ep.date()}',
                   'day_ahead_episode': self.da_price_ep,
                   'cost_history': self.reward_evol,
                   'penalty_ev_evol': self.penalty_ev_evol,
                   'line_load': [array.tolist() for array in self.line_load_evol],
                   'p_ess_filledevol':  self.p_ess_filledevol,
                   'p_cs_filledevol': self.p_cs_filledevol
                   }

        # resets lists, so they are empty for next episode to be stored
        self.reset_evol_arrays()
        results_json_ready = ndarray_to_list(results)

        # Save results as JSON
        with open(os.path.join(self.run_folder, f'results_{self.ep_count}.json'), 'w') as json_file:
            json.dump(results_json_ready, json_file)

        ##########################
        # Save the initial_values
        initial_values = {
            'soc_cs': self.init_values_cs_ep['soc_cs'],
            'soc_ess': self.init_values_ess_ep['soc_ess'],
            'arrival_t': self.init_values_cs_ep['arrival_t'],
            'evolution_of_cars': self.init_values_cs_ep['evolution_of_cars'],
            'departure_t': self.init_values_cs_ep['departure_t'],
            'present_cars': self.init_values_cs_ep['present_cars']
        }

        # Convert ndarray to list
        initial_values_json_ready = ndarray_to_list(initial_values)
        with open(os.path.join(self.run_folder, f'Initial_Values_{self.ep_count}.json'), 'w') as json_file:
            json.dump(initial_values_json_ready, json_file)

    def reset_evol_arrays(self):
        self.reward_evol = []
        self.ext_grid_evol = []
        self.pv_wasted_evol = []
        self.penalty_ev_evol = []
        self.penalty_lines_evol = []
        self.ess_power_evol = []
        self.cs_power_evol = []
        self.bus_energies_evol = []
        self.cost_ext_grid_evol = []
        self.line_load_evol = []
        self.p_ess_filledevol = []
        self.p_cs_filledevol = []

    def check_observation_size(self, dict_obs_n, obs_cs_ess_state, obs_pv, obs_load, obs_price):
        if len(obs_price) != dict_obs_n['obs_price_n']:
            print("Mismatch in obs_price. Expected:", dict_obs_n['obs_price_n'], "Actual:", len(obs_price))

        if len(obs_pv) != dict_obs_n['obs_pv_n']:
            print("Mismatch in obs_pv. Expected:", dict_obs_n['obs_pv_n'], "Actual:", len(obs_pv))

        if len(obs_load) != dict_obs_n['obs_load_n']:
            print("Mismatch in obs_load. Expected:", dict_obs_n['obs_load_n'], "Actual:", len(obs_load))

        if len(obs_cs_ess_state) != dict_obs_n['obs_cs_ess_states_n']:
            print("Mismatch in obs_cs_ess_state. Expected:", dict_obs_n['obs_cs_ess_states_n'], "Actual:", len(obs_cs_ess_state))
