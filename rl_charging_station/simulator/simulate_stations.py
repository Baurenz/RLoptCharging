import numpy as np


# todo: simulate station maybe is not the appropriate function name
def simulate_station(self):
    # arrival = self.init_values['arrival_t']
    departure = self.init_values_cs['departure_t']
    present_cars = self.init_values_cs['present_cars']
    n_cs = self.n_cs
    n_ess = self.n_ess
    # day = self.day
    timestep = self.timestep

    # calculation of which cars depart now
    leave = []
    # TODO: create more general version. now its still hour etc. think in timesteps. are big changes needed?

    if timestep <= self.last_timestep:
        for cs in range(n_cs):
            departure_car = departure[cs]
            if present_cars[cs, timestep] == 1 and (timestep + 1 in departure_car):
                leave.append(cs)

    # Generates a list indicating the timesteps until departure for each cs, with -1 for absent cars and 0 for cars departing this timestep
    state_dep_soc_cs = []
    for cs in range(n_cs):
        if present_cars[cs, timestep] == 0:
            # Append -1 to indicate no cs is present, followed by a dummy SoC value (e.g., 0)
            state_dep_soc_cs.append(-1)
            state_dep_soc_cs.append(-1)  # Dummy SoC value when no cs is present
        else:
            for departure_time in departure[cs]:
                if timestep < departure_time:
                    # Append the normalized departure timestep and the SoC for the cs
                    state_dep_soc_cs.append(round((departure_time - timestep) / self.last_timestep, 5))
                    state_dep_soc_cs.append(self.soc_cs[cs, timestep])
                    break

    state_soc_ess = []
    for ess in range(n_ess):
        state_soc_ess.append(self.soc_ess[ess, timestep])

    return leave, state_dep_soc_cs, state_soc_ess
