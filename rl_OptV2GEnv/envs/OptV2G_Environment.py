import json
from abc import ABC

import numpy as np
import os
import gymnasium as gym
from gymnasium.utils import seeding

from rl_OptV2GEnv import data_helper
from rl_OptV2GEnv.envs.env_helper_Class import CsEnvHelper
from rl_OptV2GEnv.envs.simulator import simulate_actions_charging, simulate_stations

from rl_OptV2GEnv.components import bus

from rl_OptV2GEnv.envs import reward_calc

from network_env import helper_functions

import datetime
import time


class OptV2GEnv(gym.Env, CsEnvHelper, ABC):
    # TODO: simnet = 1 & pp_network_path=None doesnt really match, think of something smart!!
    def __init__(self, price=1, solar=1, simnet=1, car_type="Car1", pp_network_path=None, network_config_path=None, debug_flag=False,
                 objective_weights=None, session_name="default", eval_flag=False, use_irradiance=True, use_real_load=True):
        super().__init__()
        self.debug_flag = debug_flag
        self.eval_flag = eval_flag
        self.done = False

        self.objective_weights = objective_weights

        self.eval_date = datetime.datetime(2022, 1, 15)

        self.penalty_ev_evol = None
        self.pv_wasted_evol = None
        self.ext_grid_evol = None
        self.p_ess_filledevol = None
        self.p_cs_filledevol = None

        self.rew_penal_dict = {}

        # TODO maybe i could have a energy_price class to handle all calculations corresponding to this
        self.da_price_raw = []  # stores price data for whole training
        self.da_price_ep = []  # stores price data for current episode
        self.da_price_ep_norm = []  # stores (min-max-normalized) price data for current episode
        self.da_price_range = None  # stores price range to be applied on reward

        self.date_ep = None  # stores current date of episode

        self.irradiance_flag = use_irradiance  # if True, uses irradiance data of random_date_episode to create pv-profiles
        self.irr_temp_cloud_raw = []  # stores irradiance_temp_cloudopacity for whole training
        self.irr_temp_cloud_ep = []  # stores irradiance_temp_cloudopacity for current episode

        self.real_load_flag = use_real_load  # if True, uses real load data of random_date_episode for load profiles
        self.real_load_raw = []  # stores load data for whole training
        self.load_data_ep = []  # stores load data for current episode

        self.np_random = None  # for seed random generator.

        # TODO: get straight where reset function is called the first time. its slightly confusing this way rn
        self.ep_count = -1
        self.day_count = -1  # TODO: only transitionally, for creation of new profiles

        self.timestep = 0  # current time step in training
        self.last_timestep = 97  # last time step of each episode
        self.total_timesteps = 0  # counting total timesteps
        self.number_of_days = 1  # counting days / episoded

        self.last_reset_time = None  # Initialize last_reset_time to None

        self.simnet_flag = simnet  # i
        self.net = None
        self.idx_dict = {}

        self.soc_cs = None
        self.soc_ess = None
        self.init_values_cs_ep = None
        self.init_values_ess_ep = None

        ######
        # storing different evolving values of episode or training? TODO
        self.reward_evol = None
        self.ess_power_evol = []


        self.leave = 0

        #####################################
        # Load configurations and Raw Data (Full Dataset)
        #####################################
        self.network_config_path = network_config_path
        self.network_config, self.n_bus, self.n_pv, self.n_load, self.n_ess, self.loc_ess, self.n_cs, self.loc_cs = \
            helper_functions.load_network_config(network_config_path)

        self.da_price_raw = data_helper.get_day_ahead_price_raw()
        self.irr_temp_cloud_raw = data_helper.get_irr_temp_cloud_raw()
        self.real_load_raw = data_helper.get_real_load_raw()

        ######################################################################
        # creates a bus object for each bus, all of them stored in a bus_dict
        ####################################################################
        self.bus_dict = {}
        for i in range(self.n_bus):
            self.bus_dict[i] = bus.Bus(bus_id=i, last_timestep=self.last_timestep, network_config_path=network_config_path)

        # # Calculate the size and start indices for each action space
        # if self.n_cs > 0:
        #     self.cs_cap_kWh = [bus.cs.cs_cap_kWh for bus in self.bus_dict.values() if bus.has_cs]
        #     self.cs_p_max = [bus.cs.cs_p_max for bus in self.bus_dict.values() if bus.has_cs]
        #
        # if self.n_ess > 0:
        #     self.ess_cap_kWh = [bus.ess.ess_cap_kWh for bus in self.bus_dict.values() if bus.has_ess]
        #     self.ess_p_max = [bus.ess.ess_p_max for bus in self.bus_dict.values() if bus.has_ess]

        ######################
        # create action space
        ####################

        # creates indexes to be able to easier separate cs and ess actions. TODO maybe redundant with n_cs and n_ess tbh
        # instance.total_action_size = 0
        self.cs_action_start_idx = None
        self.ess_action_start_idx = None

        self.action_space, self.cs_action_start_idx, self.ess_action_start_idx = self.create_action_space()

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
        # amount of observations:
        # pv: current + n_pred_pv »predictions« | price: current + n_pred_price »predictions« | car: boc + time until next departure

        self.n_pred_pv = 0  # number of pv predictions
        self.n_pred_price = 0  # number of price predictions

        self.observation_space, self.dict_obs_n = self.create_observation_space()

        if not self.eval_flag:
            # set path to store real simulation results every X episodes during training
            self.results_folder = os.path.realpath(
                os.path.join(os.path.dirname(__file__), '../..', 'solvers', 'train', 'training_results_continious'))
        else:
            # set path to store real simulation results every X episodes during evaluation
            self.results_folder = os.path.realpath(
                os.path.join(os.path.dirname(__file__), '../..', 'solvers', 'evaluation', 'Results'))

        self.run_folder = os.path.join(self.results_folder, f"{session_name}")

        try:
            os.makedirs(self.run_folder)
        except FileExistsError:
            pass

        # TODO: maybe the whole simulation could also be part of some kind of class. e.g. part of the bus system?
        # Network simulation in pandapower when simnet_flag: True
        if self.simnet_flag:
            self.net = helper_functions.load_pp_network(pp_network_path, self.bus_dict)
            self.line_load_evol = []

        # self.idx_dict
        #############################################################
        # initializing lists to store evolving values for evaluation
        self.reset_evol_arrays()

    #####################################
    # STEP FUNCTION
    #####################################
    def step(self, actions):

        ################################
        # calculates outcome of actions
        # TODO: the action result dict is a bit weird. maybe i should work more within the bus class!
        action_results = simulate_actions_charging.simulate_actions(self, actions)

        #####################
        # calculate rewards
        reward, self.rew_penal_dict, buying_cost_ext_grid, selling_cost_ext_grid = reward_calc.calculate_reward(self, action_results)

        # TODO: i think part actually i use only for every 200 episodes or evaluation when saving results.
        #  is it stupid to do it in every step then. might be unnecessary costly.??
        if self.eval_flag or self.ep_count % 200 == 0:
            self.store_evol(action_results, reward, buying_cost_ext_grid, selling_cost_ext_grid)

        self.timestep += 1
        self.total_timesteps += 1

        if self.timestep == self.last_timestep:
            self.done = True
            if self.eval_flag or self.ep_count % 200 == 0:
                decision_result_dict = self.save_decision_results()
                self.save_every_x_init_result(decision_result_dict)

        observation = self._get_obs()

        info = self._get_info()

        if self.done:
            info['is_terminal'] = True
            info['episode_length'] = self.timestep

        return observation, -reward, self.done, False, info

    #####################################
    # RESET FUNCTION
    #####################################
    def reset(self, seed=None, options=None, reset_flag=0):

        current_time = time.time()
        if self.last_reset_time is not None and self.timestep != 0:
            elapsed_time = current_time - self.last_reset_time
            average_step_time = elapsed_time / self.last_timestep
            print(f"Episode Count: {self.ep_count}")
            print(f"Day Count: {self.day_count}")
            print(f"Average time per step: {average_step_time:.6f} seconds")

        self.ep_count += 1  # TODO: rn now I am confused where to count the next episode..
        self.day_count += 1
        self.last_reset_time = current_time
        self.timestep = 0
        self.done = False

        if self.day_count == 216:  # TODO: actually not really needed, only if power profile data loaded from csv files. maybe EOL
            # TODO: implement logic
            self.day_count = 0

        ####################
        # Load Episode Data
        ##################
        if reset_flag == 0:

            self.date_ep = data_helper.get_random_date_ep()
            # get day-ahead-price for the episode, also normalized and its range
            self.load_data_ep = data_helper.get_ep_load_data(self.real_load_raw, self.date_ep)
            self.irr_temp_cloud_ep = data_helper.get_ep_irr_temp_cloud(self.irr_temp_cloud_raw, self.date_ep)

            # get pv and load data of bus for new episode (day)
            for bus in self.bus_dict.values():
                bus.get_ep_power_data(self, self.irr_temp_cloud_ep, self.load_data_ep, self.ep_count)

            # creates initial values for charging stations and ess
            [soc_cs, arrival_t, departure_t, evolution_of_cars, present_cars] = data_helper.init_cs_pres_soc(self, self.ep_count)
            soc_ess = data_helper.reset_init_ess_soc(self, self.ep_count)

        # TODO specify the perfect evaluation scenario. could be equal more less through my whole thesis wdyt?
        elif reset_flag == 1:
            # path to evaluation data for episode:
            init_data_path = './data_eval/Initial_Values_1000.json'
            result_data_path = './data_eval/results_1000.json'  # result might be misleading but contains pv, load data etc.

            with open(init_data_path, 'r') as file:
                init_data_dict = json.load(file)

            # Read and store the results data in a dictionary
            with open(result_data_path, 'r') as file:
                results_data_dict = json.load(file)

            # same for every episode (to check consistency and to compare different models against each other under the same c
            self.bus_dict = data_helper.get_ep_bus_data_eval(self.bus_dict, results_data_dict)
            self.date_ep = self.eval_date
            # TODO NEXT
            [soc_cs, arrival_t, departure_t, evolution_of_cars, present_cars] = data_helper.eval_init_cs_pres_soc(self.n_cs, init_data_dict)

            soc_ess = data_helper.eval_init_ess_soc(self.n_ess, init_data_dict)

        elif reset_flag == 2:
            self.date_ep = data_helper.get_eval_date_ep(self.ep_count)
            self.load_data_ep = data_helper.get_ep_load_data(self.real_load_raw, self.date_ep)
            self.irr_temp_cloud_ep = data_helper.get_ep_irr_temp_cloud(self.irr_temp_cloud_raw, self.date_ep)

            for bus in self.bus_dict.values():
                bus.get_ep_power_data(self, self.irr_temp_cloud_ep, self.load_data_ep, self.ep_count)

            [soc_cs, arrival_t, departure_t, evolution_of_cars, present_cars] = data_helper.init_cs_pres_soc(self, self.ep_count)
            soc_ess = data_helper.reset_init_ess_soc(self, self.ep_count)

        self.da_price_ep, self.da_price_ep_norm, self.da_price_range = data_helper.get_ep_price_data(self.da_price_raw, self.date_ep)

        self.init_values_cs_ep = {'soc_cs': soc_cs, 'arrival_t': arrival_t,
                                  'evolution_of_cars': evolution_of_cars,
                                  'departure_t': departure_t, 'present_cars': present_cars}

        self.init_values_ess_ep = {'soc_ess': soc_ess}

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
            self.soc_cs = self.init_values_cs_ep['soc_cs']
            self.soc_ess = self.init_values_ess_ep["soc_ess"]

        # normalized current timestep.
        obs_timestep = self.timestep / self.last_timestep

        # Calculate sine and cosine of day of year for cyclical time encoding
        # obs_day_of_year = data_helper.get_cyclical_day_of_year(self.date_ep)

        ###################################################################
        # Get CS Observations of car departure, soc & Observation of ESS soc
        # combining observations for ess and cs
        # obs_dep_soc_cs: [departure_time_1, SoC_1, ..., departure_time_n, SoC_n] for each car.
        self.leave, obs_dep_soc_cs, obs_soc_ess = simulate_stations.simulate_station(self)

        obs_cs_ess_state = np.concatenate(([np.array(obs_dep_soc_cs)] if self.n_cs > 0 else []) +
                                          ([np.array(obs_soc_ess)] if self.n_ess > 0 else []), axis=None)

        #######################################################
        # Get PV Observations of current time-step / 'Predictions' - range defined by n_pred_pv
        # Gets the real PV-Power data for the current time step and the next n_pred_pv time steps
        pv_obs_list = []
        for i in self.bus_dict:
            if self.bus_dict[i].has_pv:
                # Get the renewable production data for the current timestep and the next three timesteps
                bus_renewable_data = self.bus_dict[i].pv.pv_prod_episode[self.timestep:self.timestep + 1 + self.n_pred_pv]
                pv_obs_list.extend(bus_renewable_data)

        max_value = 15
        obs_pv = np.array(pv_obs_list) / max_value

        #######################################################
        # Get Load Observations of current time-step
        load_obs_list = []
        for i in self.bus_dict:
            if self.bus_dict[i].has_load:
                bus_load_data = self.bus_dict[i].load.load_episode[self.timestep]
                load_obs_list.append(bus_load_data)

        max_value = 5  # Total number of load values in the dataset: 2600064 | Number of load values below 5 kW: 2597303
        obs_load = np.array(load_obs_list) / max_value

        #######################################################
        # Get Price Observations of current time-step / 'Predictions' - range defined by n_pred_price
        obs_price = np.concatenate(
            (np.array([self.da_price_ep_norm[self.timestep]]),
             self.da_price_ep_norm[self.timestep + 1:self.timestep + 1 + self.n_pred_price]), axis=None)

        ##################
        # all Observations obs_day_of_year
        # obs_day_of_year

        observations = np.concatenate((obs_timestep, obs_cs_ess_state, obs_pv, obs_load, obs_price), axis=None)
        if self.ep_count == 0:
            self.check_observation_size(self.dict_obs_n, obs_cs_ess_state, obs_pv, obs_load, obs_price)
        observations = observations.astype(np.float32)

        return observations

    #####################################
    # SEED FUNCTION
    #####################################
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #####################################
    # CLOSE FUNCTION
    #####################################
    def close(self):
        # TODO whats to implement in here? could it be of great help??
        return 0

    def return_partial_costs(self):

        return self.rew_penal_dict
