import numpy as np


class RBC:
    def __init__(self, env):
        # Initialize RBC with dynamic action space
        self.n_cs = env.n_cs
        self.n_ess = env.n_ess  # not clear if needed, depends if i also want to control ess by rbc

        self.action_space = self.n_cs + self.n_ess

        # Assuming a discrete action space, adjust as necessary for continuous spaces

    def select_action(self, obs):
        actions = np.zeros(self.action_space)

        for cs in range(self.n_cs):

            # the departure hour for every spot is placed on the last 10 positions in states vector(10 spots)
            # have in mind that departure time is normalized in [0,1] so if T_leave is within the next 3 hours then
            # action[car]=1, else action[car]=solar_radiation or action[car]={mean value of solar radiation and the predicted one hour radiation}

            # There is 2x as many cs states as cs, it's the first observations
            # state_dep_soc_cs: [departure_time_1, SoC_1, ..., departure_time_n, SoC_n] for each car.

            if obs[cs * 2] == -1:
                actions[cs] = 0
            elif obs[cs * 2] > 0:
                actions[cs] = 1
            # else:
            #     # solar ratiation is states[0] and the predictions on ratiation are states[2],states[3],states[4]
            #
            #     # this case describes that if T_leave> 3 hours, then scenario 1: action is equal to the radiation
            #     # scenario 2: action is equal to the mean value of current radiation and its next hour prediction
            #
            #     # scenario 1, current value of radiation
            #     # action[car]=states[0]
            #
            #     # scenario 2, mean value of current radiation and one hour ahead
            #     action[car] = (states[0] + states[2]) / 2

        return actions