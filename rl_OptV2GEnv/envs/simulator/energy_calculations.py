from pathlib import Path

base_dir = Path('/').expanduser()
sum_pv_available_48 = 0


def compute_bus_powers(instance, timestep):
    bus_energies = {}

    # For all other bus elements the signing is based on the consumer viewpoint (positive active power means power consumption):
    # https://pandapower.readthedocs.io/en/latest/about/units.html

    for bus_id, bus in instance.bus_dict.items():

        net_power_bus, from_grid_bus, to_grid_bus, pv_available_bus, cs_from_grid = bus.calculate_net_power_bus(timestep)

        bus_energies[bus_id] = {
            'net_power_bus': net_power_bus,
            'from_grid': from_grid_bus,
            'to_grid': to_grid_bus,
            'pv_available': pv_available_bus,
            'cs_from_grid': cs_from_grid,
        }

    return bus_energies


def sum_network_energy_no_loss(instance, bus_energies):
    total_grid_energy = sum(bus_data['from_grid'] for bus_data in bus_energies.values())
    total_pv_available = sum(bus_data['pv_available'] for bus_data in bus_energies.values())
    total_net_energy = sum(bus_data['net_energy_bus'] for bus_data in bus_energies.values())

    return total_grid_energy, total_pv_available, total_net_energy


def compute_network_power(instance, bus_energies):

    timestep = instance.timestep
    total_grid_energy = sum(bus_data['from_grid'] for bus_data in bus_energies.values())
    total_pv_available = sum(bus_data['pv_available'] for bus_data in bus_energies.values())
    total_net_energy = sum(bus_data['net_power_bus'] for bus_data in bus_energies.values())

    # TODO: check what i need this for!
    total_pv_prod_timestep = sum(
        bus.pv.pv_prod_episode[timestep] for bus in instance.bus_dict.values() if bus.has_pv) if instance.n_pv > 0 else 0
    total_load_timestep = sum(bus.load.load_episode[timestep] for bus in instance.bus_dict.values()) if instance.n_load > 0 else 0
    total_p_cs = sum(bus.cs.p_cs_t for bus in instance.bus_dict.values() if bus.has_cs)
    total_p_ess = sum(bus.ess.p_ess_t for bus in instance.bus_dict.values() if bus.has_ess)
    network_energy_timestep = total_load_timestep + total_p_cs + total_p_ess - total_pv_prod_timestep

    return total_grid_energy, total_pv_available, total_net_energy, network_energy_timestep


def print_bus_energies(bus_id, timestep, net_energy_bus, pv_prod_bus, load_bus, p_cs_filled, p_ess_filled, from_grid_bus,
                       to_grid_bus, pv_available_bus, network_energy_timestep):
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
    print(f"pv_available: {pv_available_bus}")
    print(f"network_energy_timestep: {network_energy_timestep}")
    print("--------------------------")
