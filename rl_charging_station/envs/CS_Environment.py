import json
from abc import ABC
from pathlib import Path

import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from scipy.io import loadmat, savemat

from rl_charging_station.envs import env_helper_functions
from rl_charging_station.simulator import energy_calculations
from rl_charging_station.simulator import simulate_actions_charging_multiple
from rl_charging_station.simulator import simulate_stations

from rl_charging_station.models import bus

from rl_charging_station.envs import reward_calc

from network_env import helper_functions

import datetime
import time


class CsEnv(gym.Env, ABC):
    # TODO: simnet = 1 & pp_network_path=None doesnt really match, think of something smart!!
    def __init__(self, price=1, solar=1, simnet=1, car_type="Car1", pp_network_path=None, network_config_path=None, debug_flag=False,
                 session_name="default", eval_flag=False, use_irradiance=False):
        super().__init__()
        self.debug_flag = debug_flag
        self.eval_flag = eval_flag
        self.eval_date = datetime.datetime(2022, 1, 15)

        self.penalty_ev_evol = None
        self.pv_wasted_evol = None
        self.ext_grid_evol = None
        self.energy = None

        self.rew_penal_dict = {}

        self.day_ahead_price_raw = []
        self.day_ahead_price_episode = []
        self.day_ahead_price_episode_norm = []
        self.random_date_episode = None

        self.irradiance_flag = use_irradiance
        self.solar_irradiance_temp_cloudopacity_raw = []
        self.solar_irradiance_temp_cloudopacity_episode = []

        self.day = None
        self.time_array = None
        self.current_time = None
        self.init_values_cs = None
        self.reward_evol = None
        self.np_random = None
        self.info = None
        self.timestep = 0  # change_96: before: None
        self.total_timesteps = 0

        self.network_config_path = network_config_path

        # TODO: get straight where reset function is called the first time. its slightly confusing this way rn
        self.episode_count = -1
        self.day_count = -1  # TODO: only transitionally, for creation of new profiles

        self.last_timestep = 97  # change_96
        self.number_of_days = 1
        self.price_flag = price
        self.solar_flag = solar

        self.last_reset_time = None  # Initialize last_reset_time to None

        # self.n_bus = 4

        self.n_cars_bus = 1
        self.simnet_flag = simnet
        self.net = None
        self.profile_data_dict = {}
        self.idx_dict = {}

        self.done = False
        self.soc_cs = 0

        self.init_values_ess = None
        self.soc_ess = None

        self.ess_power_evol = []

        self.leave = 0

        #####################################
        # Load configurations and Data
        #####################################
        self.network_config, self.n_bus, self.n_pv, self.n_load, self.n_ess, self.loc_ess, self.n_cs, self.loc_cs = \
            helper_functions.load_network_config(network_config_path)

        self.day_ahead_price_raw = helper_functions.load_day_ahead_prices()
        self.solar_irradiance_temp_cloudopacity_raw = helper_functions.load_solar_irradiance_temp_cloudopacity_raw()

        self.bus_dict = {}
        for i in range(self.n_bus):
            self.bus_dict[i] = bus.Bus(bus_id=i, solar_flag=self.solar_flag, car_type=car_type,
                                       n_cars_bus=self.n_cars_bus, n_bus=self.n_bus, last_timestep=self.last_timestep,
                                       network_config_path=network_config_path)

        # extract the time structure of one episode from pv data, useful because if periods differ it will be the same
        self.time_array = [datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').strftime('%H:%M:%S') for t in
                           self.bus_dict[0].pv_prod_raw['Unnamed: 0'][:self.last_timestep]]

        self.ev_param = self.bus_dict[0].ev_param

        self.current_folder = Path(__file__).parent.parent / 'files'

        # TODO creation of action space only in function also i can do the same for the observation space

        # Calculate the size and start indices for each action space
        if self.n_cs > 0:
            self.cs_cap_kWh = [bus.cs['max_e_mwh'] * 1000 for bus in self.bus_dict.values() if bus.has_cs]
            self.cs_p_max = [bus.cs['p_mw'] * 1000 for bus in self.bus_dict.values() if bus.has_cs]

        if self.n_ess > 0:
            self.ess_cap_kWh = [bus.ess['max_e_mwh'] * 1000 for bus in self.bus_dict.values() if bus.has_ess]
            self.ess_p_max = [bus.ess['p_mw'] * 1000 for bus in self.bus_dict.values() if bus.has_ess]

        # Create a single, flat action space

        self.total_action_size = 0
        self.cs_action_start_idx = None
        self.ess_action_start_idx = None

        ######################
        # create action space
        ####################

        self.action_space, self.total_action_size, self.cs_action_start_idx, self.ess_action_start_idx = env_helper_functions.create_action_space(
            self)

        # amount of observations:
        # solar: current + 3 »predictions« | price: current + 3 »predictions« | car: boc + time until next departure
        # TODO: for now number of PV equals number of buses, might be different in more complex scenarios
        # TODO/QUESTION: is pandapower simulation part of observation space or only reward function?
        #  -> should be observation as well!

        # TODO: reactivate observation space for ess
        # TODO: revise observatioon_space:
        #        - add observations for pandapower network
        #        - maybe add observations for soc
        #        - think of meaning full observations

        ###########################
        # create observation space
        #########################

        self.n_pred_pv = 0
        self.n_pred_price = 0

        self.observation_space, self.dict_obs_n = env_helper_functions.create_observation_space(self)
        # TODO: find out why seed was in original code at all

        # not needed, is it?
        # self.seed

        if self.eval_flag == False:
            self.results_folder = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'files', 'Results'))
        else:
            self.results_folder = os.path.realpath(os.path.join(os.path.dirname(__file__), '../..', 'solvers', 'evaluation', 'Results'))

        self.run_folder = os.path.join(self.results_folder, f"{session_name}")

        try:
            os.makedirs(self.run_folder)
        except FileExistsError:
            # Directory already exists, handle as needed
            pass

        # Network simulation in pp

        # TODO: why would i load the data in bus and also here??
        # TODO: maybe the whole simulation could also be part of some kind of class. e.g. part of the bus system?
        if self.simnet_flag:
            self.net, self.idx_dict, self.profile_data_dict = helper_functions.load_pp_network(pp_network_path, self.bus_dict)

    def ndarray_to_list(data):
        if isinstance(data, dict):
            return {k: CsEnv.ndarray_to_list(v) for k, v in data.items()}
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

    #####################################
    # STEP FUNCTION
    #####################################
    def step(self, actions):
        # start_time = time.time()

        # [reward, grid, res_wasted, penalty_ev, self.boc, p_charging,
        #  net_result_dict] = \

        self.current_time = self.time_array[self.timestep]

        action_results = simulate_actions_charging_multiple.simulate_control(self, actions)

        # TODO change the dict thingy maybe, only because i switched to a result dict.

        p_cs = action_results['p_cs']
        p_ess = action_results['p_ess']
        self.soc_cs = action_results['soc_cs']
        self.soc_ess = action_results['soc_ess']

        reward, wn_penalty_lines_total, wn_penalty_ev, wn_buying_cost_ext_grid, wn_selling_cost_ext_grid, wn_penalty_sum_remaining_pv, \
            cost_sum_bus_energy, buying_cost_ext_grid, selling_cost_ext_grid, sum_penalties_os_p_cs = reward_calc.calculate_reward(self,
                                                                                                                                   action_results)

        self.rew_penal_dict = {'reward': -reward,
                               'penalty_ev': -wn_penalty_ev,
                               'penalty_lines_total': -wn_penalty_lines_total,
                               'cost_sum_bus_energy': -cost_sum_bus_energy,
                               'buying_ext_grid': -wn_buying_cost_ext_grid,
                               'selling_ext_grid': -wn_selling_cost_ext_grid,
                               'penalty_sum_remaining_pv': -wn_penalty_sum_remaining_pv,
                               'sum_penalties_os_p_cs': -sum_penalties_os_p_cs
                               }
        # if penalty_ev != 0:
        #     print(f"penalty_ev different from 0 | penalty_ev = {penalty_ev}") # TODO: delete if not necessary. oly to check!

        # TODO: whats that for? make it more meaningful
        # TODO: also add a list for history of charging power of ess
        # TODO: add evolutdion for all costs to see how they advance independently?

        if self.simnet_flag:
            self.ext_grid_evol.append(
                action_results['net_result_dict']['res_ext_grid']['p_mw'][0] * 1000)  # TODO more consistency in kw / MW
        else:
            self.ext_grid_evol.append(action_results['grid_final'])

        if p_ess: self.ess_power_evol.append(p_ess.tolist())  # TODO not too neat!!
        self.cs_power_evol.append(p_cs.tolist())
        self.pv_wasted_evol.append(action_results['res_avail'])
        self.penalty_ev_evol.append(wn_penalty_ev)
        self.penalty_lines_evol.append(wn_penalty_lines_total)
        self.reward_evol.append(reward)
        self.bus_energies_evol.append(action_results['bus_energy'])
        self.cost_ext_grid_evol.append(buying_cost_ext_grid + selling_cost_ext_grid)

        self.timestep += 1
        self.total_timesteps += 1

        conditions = self._get_obs()

        if self.timestep == self.last_timestep:
            self.done = True
            # self.timestep = 0 # TODO: check if that leads to problems, but for sure makes mores sense to reset in reset!!

            # Save every x result # TODO: needs to be checked and maybe somewhere else (function itself)

            decision_result_dict = env_helper_functions.save_decision_results(self)

            self.save_every_x_init_result(decision_result_dict)

        info = self._get_info()

        if self.done:
            info['is_terminal'] = True
            info['episode_length'] = self.timestep

        return conditions, -reward, self.done, False, info
        # TODO: i should make some sense out of False (which is a parameter for Truncation!

    #####################################
    # RESET FUNCTION
    #####################################
    def reset(self, seed=None, options=None, reset_flag=0):

        current_time = time.time()
        if self.last_reset_time is not None and self.timestep != 0:
            elapsed_time = current_time - self.last_reset_time
            average_step_time = elapsed_time / self.last_timestep
            print(f"Episode Count: {self.episode_count}")
            print(f"Day Count: {self.day_count}")

            print(f"Average time per step: {average_step_time:.6f} seconds")

        self.episode_count += 1  # TODO: rn now I am confused where to count the next episode..

        self.day_count += 1

        self.last_reset_time = current_time
        self.timestep = 0
        self.day = 1
        self.done = False

        # generate pv and load data for next set of episodes
        if self.day_count == 216:  # TODO: find some more sophisticated solution, that one is to frickelig
            # TODO: implement logic
            self.day_count = 0

        self.bus_dict[0].consumed = None
        self.energy = {'consumed': self.bus_dict[0].consumed}  # , 'renewable': renewable}

        if reset_flag == 0:
            self.random_date_episode = helper_functions.get_next_episode_date()

            self.day_ahead_price_episode = helper_functions.get_next_episode_price_data(self.day_ahead_price_raw,
                                                                                        self.random_date_episode)

            self.solar_irradiance_temp_cloudopacity_episode = helper_functions.get_next_episode_irradiance_temp_data(
                self.solar_irradiance_temp_cloudopacity_raw,
                self.random_date_episode)

            # load pv and load data for new episode (day)
            self.bus_dict = helper_functions.get_next_episode_bus_data(self.bus_dict, self.day_count, self.irradiance_flag,
                                                                       self.solar_irradiance_temp_cloudopacity_episode)

            min_price = np.min(self.day_ahead_price_episode)
            max_price = np.max(self.day_ahead_price_episode)

            if min_price <= 0:
                print(min_price)
            # Step 3: Normalize the prices
            # Avoid division by zero in case all prices in the episode are the same
            if max_price != min_price:
                self.day_ahead_price_episode_norm = (self.day_ahead_price_episode - min_price) / (max_price - min_price)
            else:
                # Handle the case where all prices are the same (e.g., by setting all normalized prices to the same value, such as 0.5)
                self.day_ahead_price_episode_norm = np.full_like(self.day_ahead_price_episode, 0.5)

            # TODO: tbh i dont really get the logic, why reset_flag zero, then having reset of values lol
            #  also I'm a bit unsure about having the calculation running in the 0 dict
            [soc_cs, arrival_t, departure_t, evolution_of_cars, present_cars] = self.bus_dict[0].reset_init_cs_presence_soc(self.n_cs)

            # TODO: to be declared as initial?
            soc_ess = self.bus_dict[0].reset_init_ess_soc(self.n_ess)

        else:
            # path to evaluation data for episode:
            init_data_path = './data_eval/Initial_Values_1000.json'
            result_data_path = './data_eval/results_1000.json'  # result might be misleading but contains pv, load data etc.

            with open(init_data_path, 'r') as file:
                initial_data_dict = json.load(file)

            # Read and store the results data in a dictionary
            with open(result_data_path, 'r') as file:
                results_data_dict = json.load(file)

            # same for every episode (to check consistency and to compare different models against each other under the same c
            self.bus_dict = helper_functions.get_episode_bus_data_eval(self.bus_dict, results_data_dict)

            self.random_date_episode = self.eval_date

            self.day_ahead_price_episode = helper_functions.get_next_episode_price_data(self.day_ahead_price_raw, self.random_date_episode)

            [soc_cs, arrival_t, departure_t, evolution_of_cars, present_cars] = self.bus_dict[0].eval_init_cs_presence_soc(self.n_cs,
                                                                                                                           initial_data_dict)

            soc_ess = self.bus_dict[0].eval_init_ess_soc(self.n_ess, initial_data_dict)

        self.init_values_ess = {'soc_ess': soc_ess}

        # TODO: self.init_values_cs and self.bus_dict[0].init_values_cars seems redundant
        self.init_values_cs = {'soc_cs': soc_cs, 'arrival_t': arrival_t, 'evolution_of_cars': evolution_of_cars,
                               'departure_t': departure_t, 'present_cars': present_cars}

        # TODO: change in a way that every bus has the data for its cars. now its all stored in one dict
        #  probably that needs more deeper adjustments, or better we have a superior Bus_Network class which will hold init values
        self.bus_dict[0].init_values_cars = {'soc_cs': soc_cs, 'arrival_t': arrival_t,
                                             'evolution_of_cars': evolution_of_cars,
                                             'departure_t': departure_t, 'present_cars': present_cars}

        self.bus_dict[0].init_values_ess = self.init_values_ess

        observation = self._get_obs()

        info = {'episode_start': True}

        return observation, info

    ################
    # INFO FUNCTION
    ##############
    def _get_info(self):
        info = self.return_partial_costs()
        return info

    #######################
    # OBSERVATION FUNCTION
    #####################
    def _get_obs(self):
        if self.timestep == 0:
            # entities which are not calculated, but known for the whole episode (pv, load, etc.) will be simply stored in the right moment
            self.reward_evol = []
            self.ext_grid_evol = []
            self.pv_wasted_evol = []
            self.penalty_ev_evol = []
            self.penalty_lines_evol = []
            self.ess_power_evol = []
            self.cs_power_evol = []
            self.bus_energies_evol = []
            self.cost_ext_grid_evol = []

            # TODO thats a weird place to put the initial soc values into self.soc_cs, right?

            self.soc_cs = self.bus_dict[0].init_values_cars['soc_cs']
            self.soc_ess = self.bus_dict[0].init_values_ess["soc_ess"]

        #  fetch observation of car departure and soc & observation of ess soc
        self.leave, obs_dep_soc_cs, obs_soc_ess = simulate_stations.simulate_station(self)

        # combining observations for ess and cs
        # state_dep_soc_cs: [departure_time_1, SoC_1, ..., departure_time_n, SoC_n] for each car.

        # normalized current timestep. TODO: is it needed to have the timestep as observation or is the agent counting anyway?
        obs_timestep = self.timestep / self.last_timestep
        obs_cs_ess_state = np.concatenate(
            ([np.array(obs_dep_soc_cs)] if self.n_cs > 0 else []) +
            ([np.array(obs_soc_ess)] if self.n_ess > 0 else []),
            axis=None
        )

        # Get the real solar energy data for the current time step and the next 3 time steps
        if self.solar_flag == 1:

            renewable_data_list = []
            # Iterate through each bus in self.bus_dict
            for i in self.bus_dict:
                # Get the renewable production data for the current timestep and the next three timesteps
                bus_renewable_data = self.bus_dict[i].pv_prod_episode[self.timestep:self.timestep + 1 + self.n_pred_pv]
                renewable_data_list.extend(bus_renewable_data)

            max_value = 15
            obs_pv = np.array(renewable_data_list) / max_value

            # TODO: can be deleted as soon as normalization always correct!
            # if np.any(renewable_data < 0) or np.any(renewable_data > 1):
            #     exceeding_values = renewable_data[(renewable_data < 0) | (renewable_data > 1)]
            #     print(f"renewable_data exceeds observation_space limits. Exceeding values: {exceeding_values}")
            #     # exit()  # Exit the program
        else:
            obs_pv = None

        load_data_list = []
        if self.n_load > 0:
            for i in self.bus_dict:
                bus_load_data = self.bus_dict[i].load_episode[self.timestep]
                load_data_list.append(bus_load_data)

        max_value = 20
        obs_load = np.array(load_data_list) / max_value

        # TODO: what is disturbance in Chargym script?
        # TODO: keeping normalized or real prices
        # obs_price = np.concatenate(
        #     (np.array([self.day_ahead_price_episode_[self.timestep]]),
        #      self.day_ahead_price_episode[self.timestep + 1:self.timestep + 4]),
        #     axis=None
        # )

        obs_price = np.concatenate(
            (np.array([self.day_ahead_price_episode_norm[self.timestep]]),
             self.day_ahead_price_episode_norm[self.timestep + 1:self.timestep + 1 + self.n_pred_price]),
            axis=None
        )

        # TODO: check if price_data fits in range {0,1}
        #  normalize price_data in the right way!
        max_value = max(self.day_ahead_price_episode)
        # obs_price = obs_price / max_value
        # TODO: can be deleted as soon as normalization always correct!
        # if np.any(obs_price < 0) or np.any(obs_price > 1):
        #     exceeding_values = obs_price[(obs_price < 0) | (obs_price > 1)]
        #     print(f"price_data exceeds observation_space limits. Exceeding values: {exceeding_values}")
        #     exit()  # Exit the program

        observations = np.concatenate((obs_timestep, obs_cs_ess_state, obs_pv, obs_load, obs_price), axis=None)

        # TODO: to be deleted only for debugging!!
        if len(obs_price) != self.dict_obs_n['obs_price_n']:
            print("Mismatch in obs_price. Expected:", self.dict_obs_n['obs_price_n'], "Actual:", len(obs_price))

        if len(obs_pv) != self.dict_obs_n['obs_pv_n']:
            print("Mismatch in obs_pv. Expected:", self.dict_obs_n['obs_pv_n'], "Actual:", len(obs_pv))

        if len(obs_load) != self.dict_obs_n['obs_load_n']:
            print("Mismatch in obs_load. Expected:", self.dict_obs_n['obs_load_n'], "Actual:", len(obs_load))

        if len(obs_cs_ess_state) != self.dict_obs_n['obs_cs_ess_states_n']:
            print("Mismatch in obs_cs_ess_state. Expected:", self.dict_obs_n['obs_cs_ess_states_n'], "Actual:", len(obs_cs_ess_state))

        if self.timestep == 2000:
            print(observations)
        return observations

    #####################################
    # SEED FUNCTION
    #####################################
    def seed(self, seed=None):
        # TODO whats to implement in here? could it be of great help??
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #####################################
    # CLOSE FUNCTION
    #####################################
    def close(self):
        # TODO whats to implement in here? could it be of great help??
        return 0

    def save_every_x_init_result(self, decision_result_dict):

        # TODO: lets also save relevant data per bus and more general maybe?!
        #  - line loading
        #  -

        if self.eval_flag or self.episode_count % 200 == 0:
            renewable_production = {i: self.bus_dict[i].pv_prod_episode for i in self.bus_dict if self.bus_dict[i].has_pv}
            load = {i: self.bus_dict[i].load_episode for i in self.bus_dict if self.bus_dict[i].has_load}

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
                       'day_ahead_date': f'{self.random_date_episode.date()}',
                       'day_ahead_episode': self.day_ahead_price_episode,
                       'cost_history': self.reward_evol,
                       'penalty_ev_evol': self.penalty_ev_evol}

            # Convert ndarray to list
            results_json_ready = CsEnv.ndarray_to_list(results)

            # TODO: rearrange this and put it in another place function this is a bit overloaded in here
            # Save results as JSON
            with open(os.path.join(self.run_folder, f'results_{self.episode_count}.json'), 'w') as json_file:
                json.dump(results_json_ready, json_file)

            # Save the initial_values
            initial_values = {
                'soc_cs': self.bus_dict[0].init_values_cars['soc_cs'],
                # TODO: might be good to really save init values, now seems like all socs of episode
                'soc_ess': self.bus_dict[0].init_values_ess['soc_ess'],
                'arrival_t': self.bus_dict[0].init_values_cars['arrival_t'],
                'evolution_of_cars': self.bus_dict[0].init_values_cars['evolution_of_cars'],
                'departure_t': self.bus_dict[0].init_values_cars['departure_t'],
                'present_cars': self.bus_dict[0].init_values_cars['present_cars']
            }

            # Convert ndarray to list
            initial_values_json_ready = CsEnv.ndarray_to_list(initial_values)
            with open(os.path.join(self.run_folder, f'Initial_Values_{self.episode_count}.json'), 'w') as json_file:
                json.dump(initial_values_json_ready, json_file)

    def return_partial_costs(self):
        # Replace with your actual cost component calculations

        return self.rew_penal_dict
