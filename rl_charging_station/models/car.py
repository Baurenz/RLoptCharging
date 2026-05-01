import json
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
from numpy import random


# NEXT implement the car class as main source for car parameters

class EV:
    def __init__(self, bus_id, car_type):
        ev_param = self.load_ev_param(bus_id, car_type)
        self.charging_effic = ev_param['charging_effic']
        self.EV_capacity = ev_param['EV_capacity']
        self.discharging_effic = ev_param['discharging_effic']
        self.charging_rate = ev_param['charging_rate']
        self.discharging_rate = ev_param['discharging_rate']

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
