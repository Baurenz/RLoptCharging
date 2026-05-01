import numpy as np


# todo: simulate station maybe is not the appropriate function name
def simulate_station(instance):
    # arrival = instance.init_values['arrival_t']
    departure = instance.init_values_cs_ep['departure_t']
    present_cars = instance.init_values_cs_ep['present_cars']
    n_cs = instance.n_cs
    n_ess = instance.n_ess
    # day = instance.day
    timestep = instance.timestep

    # calculation of which cars depart now
    leave = []

    if timestep <= instance.last_timestep:
        for cs in range(n_cs):
            departure_car = departure[cs]
            if present_cars[cs, timestep] == 1 and (timestep + 1 in departure_car):
                leave.append(cs)

    # Generates a list indicating the timesteps until departure for each cs, with -1 for absent cars and 0 for cars departing this timestep
    state_dep_soc_cs = []
    for cs in range(n_cs):
        if present_cars[cs, timestep] == 0:
            # Append 0 to indicate no cs is present, followed by a dummy SoC value (e.g., 0)
            state_dep_soc_cs.append(0)
            state_dep_soc_cs.append(0)
            state_dep_soc_cs.append(0)  # Dummy SoC value when no cs is present

        else:
            for departure_time in departure[cs]:
                if timestep < departure_time:
                    # Append the normalized departure timestep and the SoC for the cs
                    state_dep_soc_cs.append(1)
                    state_dep_soc_cs.append(round((departure_time - timestep) / instance.last_timestep, 5))
                    state_dep_soc_cs.append(instance.soc_cs[cs, timestep])
                    break

    # IMPORTANT SOC_CS has always the SoC of beginning of the timestep. departure tells us car leaves in the beginning of the timestep!

    state_soc_ess = []
    for ess in range(n_ess):
        state_soc_ess.append(instance.soc_ess[ess, timestep])

    return leave, state_dep_soc_cs, state_soc_ess
