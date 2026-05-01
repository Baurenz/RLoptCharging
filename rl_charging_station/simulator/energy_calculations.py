import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path
from datetime import datetime

base_dir = Path('~/Documents/DAI/_Thesis/git/RLoptCharging').expanduser()

sum_res_available_48 = 0


def compute_bus_powers(instance, p_cs, p_ess, timestep):
    bus_energies = {}

    # For all other bus elements the signing is based on the consumer viewpoint (positive active power means power consumption):
    # https://pandapower.readthedocs.io/en/latest/about/units.html

    total_pv_prod_timestep = sum(bus.pv_prod_episode[timestep] for bus in instance.bus_dict.values()) if instance.n_pv > 0 else 0
    total_load_timestep = sum(bus.load_episode[timestep] for bus in instance.bus_dict.values()) if instance.n_load > 0 else 0
    total_p_cs = sum(p_cs) if instance.n_cs else 0
    total_p_ess = sum(p_ess) if instance.n_ess else 0

    network_energy_timestep = total_load_timestep + total_p_cs + total_p_ess - total_pv_prod_timestep

    if instance.n_cs > 0:
        loc_cs = instance.loc_cs
        p_cs_iter = iter(p_cs)
        p_cs_filled = [next(p_cs_iter) if loc else 0 for loc in loc_cs]
        # only values for buses with charging stations, every other chargingstation will have 0
    else:
        p_cs_filled = [0] * len(instance.bus_dict)


    if instance.n_ess > 0:
        loc_ess = instance.loc_ess
        p_ess_iter = iter(p_ess)
        p_ess_filled = [next(p_ess_iter) if loc else 0 for loc in loc_ess]
    else:
        p_ess_filled = [0] * len(instance.bus_dict)

    for bus_id, bus_data in instance.bus_dict.items():
        pv_prod_bus = instance.bus_dict[bus_id].pv_prod_episode[timestep]
        load_bus = instance.bus_dict[bus_id].load_episode[timestep] if instance.n_load > 0 else 0
        net_power_bus, from_grid_bus, to_grid_bus, res_available_bus, cs_from_grid = calculate_net_power_per_bus(timestep, bus_id,
                                                                                                                 network_energy_timestep,
                                                                                                                 pv_prod_bus, load_bus,
                                                                                                                 p_cs_filled[bus_id],
                                                                                                                 p_ess_filled[bus_id])




        # TODO: started something, which might not work
        #  the way i wanted because its convinient to have all
        #  bus energies in one dict
        #  instance.bus_dict[bus_id].energy_step =
        # Inside your if statement:
        if instance.debug_flag and bus_id == 0 and timestep == 48:
            print_bus_energies(bus_id, timestep, net_power_bus, pv_prod_bus, load_bus, p_cs_filled, p_ess_filled, from_grid_bus,
                               to_grid_bus, res_available_bus, network_energy_timestep)



        bus_energies[bus_id] = {
            'net_power_bus': net_power_bus,
            'from_grid': from_grid_bus,
            'to_grid': to_grid_bus,
            'res_available': res_available_bus,
            'cs_from_grid': cs_from_grid}

    return bus_energies, network_energy_timestep


def sum_network_energy_no_loss(instance, bus_energies):
    total_grid_energy = sum(bus_data['from_grid'] for bus_data in bus_energies.values())
    total_res_available = sum(bus_data['res_available'] for bus_data in bus_energies.values())
    total_net_energy = sum(bus_data['net_energy_bus'] for bus_data in bus_energies.values())

    return total_grid_energy, total_res_available, total_net_energy


def compute_network_energy(instance, bus_energies):
    total_grid_energy = sum(bus_data['from_grid'] for bus_data in bus_energies.values())
    total_res_available = sum(bus_data['res_available'] for bus_data in bus_energies.values())
    total_net_energy = sum(bus_data['net_power_bus'] for bus_data in bus_energies.values())

    # TODO: check if I really need all of those values

    return total_grid_energy, total_res_available, total_net_energy


def calculate_net_power_per_bus(timestep, bus_id, network_energy_timestep, pv_prod_bus, load_bus, p_cs, p_ess):
    """
    Calculate the net energy for a given bus.
    """
    global sum_res_available_48

    if bus_id == 0:
        # There are three bus elements that have power values based on the generator viewpoint
        # (positive active power means power generation), which are:gen, sgen, ext_grid
        net_energy_bus = load_bus + p_cs + p_ess - pv_prod_bus  # - network_energy_timestep

    else:
        net_energy_bus = load_bus + p_cs + p_ess - pv_prod_bus
        # TODO: check if signs are correct!! --> SHOULD BE FINE: p_mw -
        #  The momentary active power of the storage (positive for charging, negative for discharging)

    # Energy needed from the grid
    from_grid_bus = max(0, net_energy_bus)
    to_grid_bus = min(0, net_energy_bus)

    # Excess renewable energy available (solely from PV production)
    res_available_bus = max(0, pv_prod_bus - (load_bus + max(0, p_cs) + max(0, p_ess)))


    # calculate the power the charging station has to take from the grid.
    total_load = load_bus + max(0, p_cs) + max(0, p_ess)  # Sum of all demands, ensuring p_cs is only added when it's a demand
    remaining_demand_after_pv = total_load - pv_prod_bus  # Subtracting PV production from total demand
    remaining_demand = remaining_demand_after_pv + min(0, p_ess)  # Adding ESS contribution (if discharging)
    cs_from_grid = min(max(0, remaining_demand), max(0, p_cs))  # `cs_from_grid` is part of the remaining demand, capped at p_cs demand


    # TODO can be deleted
    if timestep == 200:
        print(f"pv_prod_bus: {pv_prod_bus}")
        print(f"load_bus: {load_bus}")
        print(f"p_cs): {p_cs}")
        print(f"p_ess): {p_ess}")
        sum_res_available_48 += res_available_bus
        print(f"res_available): {res_available_bus}")
        if bus_id == 4:
            print(f"sum_res_available_48): {sum_res_available_48}")
            sum_res_available_48 = 0

    return net_energy_bus, from_grid_bus, to_grid_bus, res_available_bus, cs_from_grid


def calculate_energy_from_grid(instance, bus_id, bus_data):
    """
    Calculate the energy bought from the grid for a given bus.
    """

    energy_from_grid = 0
    return energy_from_grid


def print_bus_energies(bus_id, timestep, net_energy_bus, pv_prod_bus, load_bus, p_cs_filled, p_ess_filled, from_grid_bus,
                       to_grid_bus, res_available_bus, network_energy_timestep):
    print("\n---Energy Calculation-----")
    print(f"for Bus: {bus_id} @ timestep {timestep}")
    print(f"net_energy_bus: {net_energy_bus}")
    print(f"pv_prod_bus: {pv_prod_bus}")
    print(f"load_bus: {load_bus}")
    print(f"p_cs: {p_cs_filled[bus_id]}")
    print(f"p_ess: {p_ess_filled[bus_id]}")
    print("--------------------------")
    print(f"from_grid: {from_grid_bus}")
    print(f"to_grid: {to_grid_bus}")
    print(f"res_available: {res_available_bus}")
    print(f"network_energy_timestep: {network_energy_timestep}")
    print("--------------------------")
