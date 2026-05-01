import numpy as np
from network_env.helper_functions import run_single_timestep_pf


def simulate_control(self, actions):
    hour = self.timestep
    # TODO: make some sense out of consumed; so far always 0. could be used for e.g.
    #  consumption per node/bus
    consumed = self.energy['consumed']
    renewable = self.energy['renewable']
    present_cars = self.init_values_cs['present_cars']
    # print(present_cars[:,hour])

    leave = self.leave
    boc = self.soc_cs
    p_charging = np.zeros(self.number_of_cars)

    for car in range(self.number_of_cars):
        if actions[car] >= 0:
            max_charging_energy = min([10, (1 - boc[car, hour]) * self.ev_param['EV_capacity']])
        else:
            max_charging_energy = min([10, boc[car, hour] * self.ev_param['EV_capacity']])
        # in case action=[-100,100] p_charging[car] = actions[car]/100*max_charging_energy otherwise if action=[-1,1] p_charging[car] = 100*actions[car]/100*max_charging_energy

        # p_charging[car] = actions[car]/100*max_charging_energy
        # p_charging[car] = 100 * actions[car] / 100 * max_charging_energy
        # TODO: pretty sure that's not really working this way for different power ranges etc..
        if present_cars[car, hour] == 1:
            p_charging[car] = 100 * actions[car] / 100 * max_charging_energy
        else:
            p_charging[car] = 0

    # Calculation of next state of Battery based on actions
    # ----------------------------------------------------------------------------
    # TODO: I'll need something like a car class. so i can have different cars
    #  i.e. different capacity, charging limits etc.
    for car in range(self.number_of_cars):
        if present_cars[car, hour] == 1:
            boc[car, hour + 1] = boc[car, hour] + p_charging[car] / self.ev_param['EV_capacity']

    # pp_outcome = None
    if self.simnet_flag:
        net_result_dict = run_single_timestep_pf(self.net, time_step=self.timestep,
                                                 profile_data_dict=self.profile_data_dict,  # maybe i want to get rid of the profile data dict,
                                                 bus_dict=self.bus_dict,
                                                 idx_dict=self.idx_dict, cs_action_p=p_charging)
    else:
        net_result_dict = None

    # Calculation of energy utilization from the PV
    # Calculation of energy coming from Grid
    # ----------------------------------------------------------------------------
    # TODO: consumed for now is always 0
    res_avail = max([0, renewable[0, hour] - consumed[0, hour]])
    total_charging = sum(p_charging)

    # First cost index
    # ----------------------------------------------------------------------------
    # TODO: does clipping have a negative effect?
    grid_final = max([total_charging - res_avail, 0])

    # TODO: check if grid_final == net_result_dict['res_ext_grid']
    # if grid_final > 0:
    #     # print(grid_final)
    #     # print(net_result_dict['res_ext_grid'])
    #
    if self.simnet_flag:
        cost_1 = grid_final * self.energy["price"][0, hour]

        cost_ev = []
        for ii in range(len(leave)):
            cost_ev.append(((1 - boc[leave[ii], hour + 1]) * 2) ** 2)
        cost_3 = sum(cost_ev)

        # TODO now only for single line:
        if net_result_dict['res_line']['loading_percent'][0] > 0.8:
            cost_grid = 2
        else:
            cost_grid = net_result_dict['res_line']['loading_percent'][0] / 5
        cost = cost_1 + cost_3 + cost_grid

    else:
        cost_1 = grid_final * self.energy["price"][0, hour]

        # Second cost index
        # Penalty of wasted RES energy

        # res_avail = max([res_avail-total_charging, 0])
        # Cost_2 = -res_avail * (self.Energy["Price"][0, hour]/2)

        # Third cost index
        # Penalty of not fully charging the cars that leave
        # ----------------------------------------------------------------------------

        cost_ev = []
        for ii in range(len(leave)):
            cost_ev.append(((1 - boc[leave[ii], hour + 1]) * 2) ** 2)
        cost_3 = sum(cost_ev)

        cost = cost_1 + cost_3

    return cost, grid_final, res_avail, cost_3, boc, p_charging, net_result_dict
