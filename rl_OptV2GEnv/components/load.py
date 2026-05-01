import numpy as np
from numpy import random


class Load:
    """Represents a PV which belongs to a bus, including all values for PV"""

    def __init__(self):
        self.load_episode = []
        self.p_load_t = 0

    def get_real_load_episode(self, real_load_episode, bus_id, ep_count):
        rng = np.random.RandomState(bus_id + ep_count)
        # Use the RandomState instance to pick 1 of the 70 household load-profiles of that day
        random_col_index = rng.randint(0, real_load_episode.shape[1])
        # Select the column data using the random index
        self.load_episode = real_load_episode[:, random_col_index]

        # Select the column data using the random index
        self.load_episode = real_load_episode[:, random_col_index]
