import json
from pathlib import Path
from typing import Dict

from rl_OptV2GEnv.components import pv, load, cs, ess

import pandas as pd
import numpy as np


class Bus:
    """Represents a bus that has various attributes including consumption and production."""

    def __init__(self, bus_id: int, last_timestep, network_config_path):
        self.bus_id: int = bus_id
        self.last_timestep = last_timestep

        # TODO why do i have consumption as well?
        #  in original chargym consumed is only difference between pv and car charging
        #  consumption in here will be load profile of e.g. household

        # TODO: would be better to load it only once. not for every bus individually
        with open(network_config_path, 'r') as file:
            network_config = json.load(file)

        # Access the configuration for this specific bus
        bus_config = network_config['components'][f'bus{self.bus_id}']

        # Set boolean values based on existence of components
        self.has_ext_grid = None  # TODO: implement storing ext_grid
        self.has_pv = 'pv' in bus_config
        self.has_load = 'load' in bus_config
        self.has_ess = 'ess' in bus_config
        self.has_cs = 'cs' in bus_config

        if self.has_load:
            self.load = load.Load()

        if self.has_pv:
            self.pv_kWp = bus_config['pv']['kWp']
            self.pv = pv.PV(bus_config['pv']['kWp'])

        if self.has_cs:
            self.cs_values = bus_config['cs']
            self.cs = cs.CS(self.cs_values)

        if self.has_ess:
            self.ess_values = bus_config['ess']
            self.ess = ess.Ess(self.ess_values)

        #################(
        # episode / timestep values
        ###############
        self.p_from_grid_t = None
        self.p_to_grid_t = None
        self.cs_from_grid_t = None

        self.pv_available_t = None
        self.p_net_t = None

        #####
        # reward values
        self.grid_cost_t = None
        self.grid_revenue_t = None

        ### TODO: EOL / only needed if profiles loaded from files:
        self.pv_prod_raw: int = 0
        self.load_raw: int = 0

    def get_ep_power_data(self, instance, irr_temp_cloud_ep, load_data_episode, ep_count):

        # TODO: else condition for irradiance_flag would apply for loading pv, load profiles from csv (old approach, could be reactivated?)
        #   check commented function data_helper.get_next_episode_bus_data()
        if self.has_pv:
            if instance.irradiance_flag:
                self.pv.get_pv_prod_ep_ambient(irr_temp_cloud_ep, self.bus_id, ep_count)

        if self.has_load:
            if instance.real_load_flag:
                self.load.get_real_load_episode(load_data_episode, self.bus_id, ep_count)

    @staticmethod
    def load_load(bus_id: int) -> int:
        # TODO: EOL!
        """Loads the pv consumption data for the bus for 209 days ~ 20000 timeesteps. """

        file_name = f'data/dynamic_profiles/20000/bus{bus_id}_load.csv'
        consumption_raw = pd.read_csv(Path(file_name))
        # csv is in MW we convert it to kW
        consumption_raw[f'bus{bus_id}_load'] *= 1000

        return consumption_raw

    def get_load_episode(self, day_count):
        # TODO: EOL!
        """since load_consumption will load 209 days (enough data for 20000 timesteps)
        here data for each episode will be loaded """

        start_index = day_count * 102
        end_index = (day_count + 1) * 102
        bus_id = self.bus_id

        load_key = f'bus{bus_id}_load'
        if load_key in self.load_raw:
            self.load.load_episode = self.load_raw[load_key].values[start_index:end_index]

        return self.load.load_episode

    @staticmethod
    def load_pv_production(bus_id):
        """Loads the pv production data for the bus for 209 days ~ 20000 timeesteps. """

        file_name = f'data/dynamic_profiles/20000/bus{bus_id}_pv.csv'
        pv_production_raw = pd.read_csv(Path(file_name))
        # csv is in MW we convert it to kW
        pv_production_raw[f'bus{bus_id}_pv'] *= 1000

        return pv_production_raw

    def get_pv_production_episode(self, episode_count):
        """since load_pv_production will load 209 days (enough data for 20000 timesteps)
        here data for each episode will be loaded """

        start_index = episode_count * 102
        end_index = (episode_count + 1) * 102

        pv_key = f'bus{self.bus_id}_pv'
        if pv_key in self.pv_prod_raw:
            bus_pv_prod_episode = self.pv_prod_raw[pv_key].values[start_index:end_index]

        return bus_pv_prod_episode


    def calculate_net_power_bus(self, timestep):
        """
        Calculate the net energy for a given bus.
        """
        global sum_pv_available_48

        pv_prod_t = self.pv.pv_prod_episode[timestep] if self.has_pv else 0
        load_bus_t = self.load.load_episode[timestep] if self.has_load else 0
        p_cs_t = self.cs.p_cs_t if self.has_cs else 0
        p_ess_t = self.ess.p_ess_t if self.has_ess else 0

        if self.bus_id == 0:
            # There are three bus elements that have power values based on the generator viewpoint
            # (positive active power means power generation), which are:gen, sgen, ext_grid
            self.p_net_t = load_bus_t + p_cs_t + p_ess_t - pv_prod_t  # - network_energy_timestep

        else:
            self.p_net_t = load_bus_t + p_cs_t + p_ess_t - pv_prod_t
            # TODO: check if signs are correct!! --> SHOULD BE FINE: p_mw -
            #  The momentary active power of the storage (positive for charging, negative for discharging)

        self.p_from_grid_t = max(0, self.p_net_t)
        self.p_to_grid_t = min(0, self.p_net_t)

        # Excess renewable energy available (solely from PV production)
        self.pv_available_t = max(0, pv_prod_t - (load_bus_t + max(0, p_cs_t) + max(0, p_ess_t)))

        # calculate the power the charging station has to take from the grid.
        total_load = load_bus_t + max(0, p_cs_t) + max(0, p_ess_t)  # Sum of all demands, ensuring p_cs is only added when it's a demand
        remaining_demand_after_pv = total_load - pv_prod_t  # Subtracting PV production from total demand
        remaining_demand = remaining_demand_after_pv + min(0, p_ess_t)  # Adding ESS contribution (if discharging)
        self.cs_from_grid_t = min(max(0, remaining_demand),
                                  max(0, p_cs_t))  # `cs_from_grid` is part of the remaining demand, capped at p_cs demand

        # TODO can be deleted
        if timestep == 200:
            print(f"pv_prod_t: {pv_prod_t}")
            print(f"load_bus_t: {load_bus_t}")
            print(f"p_cs): {p_cs_t}")
            print(f"p_ess): {p_ess_t}")
            sum_pv_available_48 += self.pv_available_t
            print(f"pv_available): {self.pv_available_t}")
            if self.bus_id == 4:
                print(f"sum_pv_available_48): {sum_pv_available_48}")
                sum_pv_available_48 = 0

        return self.p_net_t, self.p_from_grid_t, self.p_to_grid_t, self.pv_available_t, self.cs_from_grid_t



    # @staticmethod
    # def load_ev_param(bus_id, car_type):
    #     """Loads the car parameters production data for the bus. """
    #     # TODO for now only one car type for all of the cars. bc probably adjustments needed to handle different cars
    #     # TODO: depending on implementation details, might be useful to have own class for cars
    #     #  e.g. car could be enabled to switching station etc
    #
    #     car_parameters_path = 'data/scenarios/car_parameters.json'
    #     with open(car_parameters_path) as json_file:
    #         car_data = json.load(json_file)
    #
    #     # TODO: for now only one car type for all of the cars. bc probably adjustments needed to handle different cars
    #     car_param = car_data[car_type]
    #
    #     ev_capacity = car_param["EV_capacity"]
    #     charging_effic = car_param["charging_effic"]
    #     discharging_effic = car_param["discharging_effic"]
    #     charging_rate = car_param["charging_rate"]
    #     discharging_rate = car_param["discharging_rate"]
    #     ev_param = {'charging_effic': charging_effic, 'EV_capacity': ev_capacity,
    #                 'discharging_effic': discharging_effic, 'charging_rate': charging_rate,
    #                 'discharging_rate': discharging_rate}
    #
    #     return ev_param
