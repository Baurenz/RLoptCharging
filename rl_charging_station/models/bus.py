import json
from pathlib import Path
from typing import Dict

import pandas as pd

import numpy as np
from numpy import random


class Bus:
    """Represents a bus that has various attributes including consumption and production."""

    def __init__(self, bus_id: int, solar_flag: bool, car_type, n_cars_bus, n_bus, last_timestep, network_config_path):
        self.bus_id: int = bus_id
        self.n_cars_bus = n_cars_bus
        self.n_bus = n_bus

        self.last_timestep = last_timestep
        # TODO: info to be implemented
        self.charge_station_info: Dict = {}
        self.solar_flag = solar_flag

        # TODO: not sure if dict for energy needed, maybe can be covered by variables
        self.energy = {}
        self.pv_prod_raw: int = 0
        self.pv_prod_episode = []

        self.load_raw: int = 0
        self.load = []
        self.load_episode = []

        self.energy_step = {}

        # TODO why do i have consumption as well?
        #  in original chargym consumed is only difference between pv and car charging
        #  consumption in here will be load profile of e.g. household

        with open(network_config_path, 'r') as file:
            network_config = json.load(file)

        # Access the configuration for this specific bus
        bus_config = network_config['components'][f'bus{bus_id}']

        # Set boolean values based on existence of components
        self.has_ext_grid = None  # TODO: implement storing ext_grid
        self.has_pv = 'pv' in bus_config
        self.has_load = 'load' in bus_config
        self.has_ess = 'ess' in bus_config
        self.has_cs = 'cs' in bus_config
        self.has_v2g = self.has_cs and bus_config['cs'].get('v2g', False)

        if self.has_cs:
            self.cs = {'p_mw': bus_config['cs']['p_mw'],
                       'max_e_mwh': bus_config['cs']['max_e_mwh']
                       }
        if self.has_pv:
            self.pv_kWp = bus_config['pv']['kWp']

        if self.has_ess:
            self.ess = {'p_mw': bus_config['ess']['p_mw'],
                        'max_e_mwh': bus_config['ess']['max_e_mwh'],
                        'soc': bus_config['ess']['soc']}

        # TODO: init values for all buses, will be stored in 'bus_dict.bus[0]', does it make sense
        self.init_values_cars = {}

        self.init_values_ess = {}

        # Load data from files:
        self.load_raw = self.load_load(self.bus_id)

        if self.solar_flag == 1:
            self.pv_prod_raw = self.load_pv_production(self.bus_id)

        self.ev_param = self.load_ev_param(bus_id, car_type=car_type)

    @staticmethod
    def load_load(bus_id: int) -> int:
        """Loads the pv consumption data for the bus for 209 days ~ 20000 timeesteps. """

        file_name = f'../../data/dynamic_profiles/20000/bus{bus_id}_load.csv'
        consumption_raw = pd.read_csv(Path(file_name))
        # csv is in MW we convert it to kW
        consumption_raw[f'bus{bus_id}_load'] *= 1000

        return consumption_raw

    def get_load_episode(self, day_count, bus_id):
        """since load_consumption will load 209 days (enough data for 20000 timesteps)
        here data for each episode will be loaded """

        start_index = day_count * 102
        end_index = (day_count + 1) * 102

        load_key = f'bus{bus_id}_load'
        if load_key in self.load_raw:
            bus_load_episode = self.load_raw[load_key].values[start_index:end_index]

        return bus_load_episode

    @staticmethod
    def load_pv_production(bus_id):
        """Loads the pv production data for the bus for 209 days ~ 20000 timeesteps. """

        file_name = f'../../data/dynamic_profiles/20000/bus{bus_id}_pv.csv'
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

    def get_pv_production_episode_irradiance(self, solar_irradiance_temp_cloudopacity_episode):
        if not self.has_pv:
            return np.zeros(len(solar_irradiance_temp_cloudopacity_episode))  # Return an array of zeros if no PV is present

        # Constants
        G_standard = 1000  # Standard solar radiation in W/m²
        T_standard = 25  # Standard cell temperature in °C
        alpha_T = -0.004  # Temperature coefficient in 1/°C
        pv_kWp = self.pv_kWp  # Rated power of the PV system in kWp

        # Initialize an array to hold the PV production values for each timestep
        pv_prod = np.zeros(len(solar_irradiance_temp_cloudopacity_episode))

        for i, (G, T_ambient, cloud_opacity) in enumerate(solar_irradiance_temp_cloudopacity_episode):
            # Calculate cell temperature for the current timestep
            T_cell = T_ambient + (T_standard / 800) * G

            # Calculate PV power output for the current timestep
            P_PV = pv_kWp * (G / G_standard) * (1 + alpha_T * (T_cell - T_standard))

            # Ensure that PV power output is not negative
            P_PV = max(P_PV, 0)

            # Determine the noise level based on cloud opacity
            if 30 <= cloud_opacity <= 70:
                noise_level = -abs(np.random.normal(0, 0.1))  # Higher variability in this cloud opacity range
            else:
                noise_level = -abs(np.random.normal(0, 0.05))  # Lower variability outside this range

            # Apply noise level to the calculated PV power output
            P_PV_noise = P_PV * (1 + noise_level)
            P_PV_noise = max(P_PV_noise, 0)  # Ensure that PV power output with noise is not negative

            # Store the calculated PV power output with noise in the pv_prod array
            pv_prod[i] = P_PV_noise

        return pv_prod
    @staticmethod
    def load_ev_param(bus_id, car_type):
        """Loads the car parameters production data for the bus. """
        # TODO for now only one car type for all of the cars. bc probably adjustments needed to handle different cars
        # TODO: depending on implementation details, might be useful to have own class for cars
        #  e.g. car could be enabled to switching station etc

        car_parameters_path = Path(__file__).parent / '../../data/scenarios/car_parameters.json'
        with open(car_parameters_path) as json_file:
            car_data = json.load(json_file)

        # TODO: for now only one car type for all of the cars. bc probably adjustments needed to handle different cars
        car_param = car_data[car_type]

        ev_capacity = car_param["EV_capacity"]
        charging_effic = car_param["charging_effic"]
        discharging_effic = car_param["discharging_effic"]
        charging_rate = car_param["charging_rate"]
        discharging_rate = car_param["discharging_rate"]
        ev_param = {'charging_effic': charging_effic, 'EV_capacity': ev_capacity,
                    'discharging_effic': discharging_effic, 'charging_rate': charging_rate,
                    'discharging_rate': discharging_rate}

        return ev_param

    def eval_init_cs_presence_soc(self, n_cs, initial_data_dict):
        # Extract data from the dictionary
        soc_cs_init = np.array(initial_data_dict['soc_cs'])  # Assuming 'boc' contains the initial state of charge for each charging station
        arrival_t = initial_data_dict['arrival_t']
        departure_t = initial_data_dict['departure_t']
        evolution_of_cars = np.array(initial_data_dict['evolution_of_cars'])
        present_cars = np.array(initial_data_dict['present_cars'])

        # Ensure the data shapes are correct, especially if the number of charging stations (n_cs) is different from what was originally saved
        if soc_cs_init.shape[0] != n_cs:
            raise ValueError(
                f"The provided initial data has a different number of charging stations ({soc_cs_init.shape[0]}) than expected ({n_cs}).")

        # No need to generate data, as we are using the provided initial_data_dict
        # Return the data in the same format as reset_init_cs_presence_soc
        return soc_cs_init, arrival_t, departure_t, evolution_of_cars, present_cars

    def reset_init_cs_presence_soc(self, n_cs):
        arrival_probabilities = [(0, 0.85), (8, 0.1), (32, 0.05), (48, 0.1), (64, 0.15), (80, 0.15), (96, 0.15)]
        departure_probabilities = [(0, 0.04), (28, 0.2), (36, 0.1), (48, 0.15), (64, 0.1), (80, 0.1), (96, 0.1)]

        soc_cs_init = - np.ones([n_cs, self.last_timestep + 1])
        present_cars = np.zeros([n_cs, self.last_timestep + 1])
        arrival_t = []
        departure_t = []

        def get_probability(probabilities, hour):
            for time, prob in reversed(probabilities):
                if hour >= time:
                    return prob
            return 0

        for car in range(n_cs):
            present = False
            arrival_car = []
            departure_car = []
            arrival_hour = None  # Variable to track the arrival hour

            for hour in range(self.last_timestep):
                if not present and hour < self.last_timestep - 2:  # Preventing arrivals in the last two timesteps
                    arrival_prob = get_probability(arrival_probabilities, hour)
                    present = random.random() < arrival_prob
                    if present:
                        soc_cs_init[car, hour] = random.randint(20, 60) / 100
                        arrival_car.append(hour)
                        arrival_hour = hour  # Set the arrival hour when the car arrives

                elif present:
                    # Check for departure only if at least 6 timesteps have passed since arrival
                    if hour >= arrival_hour + 14:
                        departure_prob = get_probability(departure_probabilities, hour)
                        will_leave = random.random() < departure_prob
                        if will_leave:
                            departure_car.append(hour)
                            present = False
                            arrival_hour = None  # Reset the arrival hour

                present_cars[car, hour] = 1 if present else 0

            # Ensure departure at the last timestep if still present
            if present:
                present_cars[car, self.last_timestep] = 0
                departure_car.append(self.last_timestep)

            arrival_t.append(arrival_car)
            departure_t.append(departure_car)

        evolution_of_cars = np.sum(present_cars, axis=0)

        # print(soc_cs_init)
        # print(arrival_t)
        # print(departure_t)
        # print(evolution_of_cars)
        # print(present_cars)

        return soc_cs_init, arrival_t, departure_t, evolution_of_cars, present_cars

    def eval_init_ess_soc(self, n_ess, initial_data_dict):
        # Initialize soc_ess with zeros for all timesteps
        soc_ess = np.zeros([n_ess, self.last_timestep + 1])

        # Load the initial SoC values from the dictionary for the first timestep
        # and ensure it's a numpy array to facilitate indexing
        initial_soc = np.array(initial_data_dict['soc_ess'])

        # Ensure that the provided initial SoC data matches the expected number of ESS units
        if initial_soc.shape[0] != n_ess:
            raise ValueError(
                f"The provided initial data has a different number of ESS units ({initial_soc.shape[0]}) than expected ({n_ess}).")

        # Fill the first timestep of soc_ess with the loaded initial SoC values
        soc_ess[:, 0] = initial_soc[:, 0]

        # The rest of soc_ess is already initialized to zeros
        return soc_ess

    def reset_init_ess_soc(self, n_ess):
        # Ensure that the limits are between 0 and 1

        lower_limit = 0.2
        upper_limit = 1

        # Initialize soc_ess with zeros for all timesteps
        soc_ess = np.zeros([n_ess, self.last_timestep + 1])

        # Generate random SoC values for the first timestep only
        soc_ess[:, 0] = np.random.uniform(lower_limit, upper_limit, n_ess)

        return soc_ess

    def generate_pv_production(self):
        # Placeholder function to regenerate PV production data
        # self.pv_production_raw = random.uniform(0, 100)
        # TODO: copy generation from create_data/create_profiles_dynamic_multipleDays.py
        #   to be called whenerver pv_production is ending.

        pass

    def generate_consumption(self):
        # Placeholder function to regenerate consumption data
        # TODO: copy generation from create_data/create_profiles_dynamic_multipleDays.py
        #   to be called whenerver pv_production is ending.
        pass
        # self.load = random.uniform(0, 50)

    def generate_charge_station_info(self):
        # Placeholder function to regenerate charging station information
        self.charge_station_info = {
            'capacity': random.randint(1, 5),
            'charging_speed': random.uniform(1, 3),
        }

    def reset(self):
        # Regenerate all data for the bus
        self.generate_pv_production()
        self.generate_charge_station_info()
        self.generate_consumption()

# # Dynamic object creation
# num_buses = 10
# bus_objects = [Bus(bus_id=i) for i in range(num_buses)]
#
# # Resetting all buses
# for bus in bus_objects:
#     bus.reset()
#
# # Print information for debugging
# for bus in bus_objects:
#     print(
#         f"Bus ID: {bus.bus_id}, PV Production: {bus.renewable_production}, Charge Station Info: {bus.charge_station_info}, Consumption: {bus.consumption}")
