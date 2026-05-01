import os
import json
import sys

import numpy as np

######################
### SESSION SELECTION
####################
model_type = 'DDPG'
session_prefix = "scenariolow"
session_id = 1


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def get_file_paths(model_type, session_prefix, session_id):
    base_path = f'Results/{model_type}_{session_prefix}{session_id}/'
    init_files = []
    result_files = []

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.startswith('Initial_Values_') and file.endswith('.json'):
                init_files.append(os.path.join(root, file))
            elif file.startswith('results_') and file.endswith('.json'):
                result_files.append(os.path.join(root, file))

    return sorted(init_files), sorted(result_files)


def load_all_files(init_files, result_files):
    init_jsons = [load_json(file) for file in init_files]
    result_jsons = [load_json(file) for file in result_files]

    return init_jsons, result_jsons


# Get file paths
init_files, result_files = get_file_paths(model_type, session_prefix, session_id)

# Load all JSON files
init_jsons, result_jsons = load_all_files(init_files, result_files)

# Now, you can proceed to extract data and perform calculations like averaging.
# For example, to calculate the average SOC for all episodes, you could do something like this:

import numpy as np


def calculate_average_soc_on_leave(result_jsons):
    average_soc_values_dep = {}  # Dictionary to hold the average SoC values for each car
    all_soc_values = []  # List to hold all SoC values for computing the overall average

    for result in result_jsons:
        cs_soc_leave = result.get('cs_soc_leave', [])

        for car_index, soc_values in enumerate(cs_soc_leave):
            if car_index not in average_soc_values_dep:
                average_soc_values_dep[car_index] = []

            average_soc_values_dep[car_index].extend(soc_values)  # Temporarily store all values
            all_soc_values.extend(soc_values)  # Add values to the overall list

    # Calculate the average SoC for each car
    for car_index, soc_values in average_soc_values_dep.items():
        average_soc_values_dep[car_index] = np.mean(soc_values)  # Replace list with average value
        print(f'Car {car_index}: Average SoC on Departure = {average_soc_values_dep[car_index]:.2f}')

    # Calculate and print the overall average SoC for all cars
    overall_average_soc_dep = np.mean(all_soc_values)
    print(f'\nOverall Average SoC on Departure for All Cars = {overall_average_soc_dep:.2f}')

    return average_soc_values_dep, overall_average_soc_dep


def calculate_net_grid_energy(result_jsons):
    net_energy_from_grid_per_bus = {}

    for result in result_jsons:
        renewable_energies = result['renewable']  # PV generation per bus
        load_energies = result['load']  # Load consumption per bus
        ess_values = result['p_ess_filledevol']  # ESS interaction values per timestep for all buses
        cs_values = result['p_cs_filledevol']  # CS interaction values per timestep for all buses

        max_timesteps = 96  # Limiting to the first 96 timesteps

        for bus_id in load_energies.keys():
            net_energy_from_grid_per_bus[bus_id] = 0

            for timestep in range(max_timesteps):
                # Calculate the net load for this timestep
                net_load = load_energies[bus_id][timestep] - renewable_energies[bus_id][timestep] - ess_values[bus_id][timestep] - \
                           cs_values[bus_id][timestep]

                # If the net load is negative (indicating surplus energy), consider it as zero
                if net_load < 0:
                    net_load = 0

                # Accumulate only the positive net load values
                net_energy_from_grid_per_bus[bus_id] += net_load /4

                print(net_energy_from_grid_per_bus)

    return net_energy_from_grid_per_bus
def calculate_total_feed_in_gain(result_jsons):
    total_feed_in_gain_per_bus = {}
    total_feed_in_power_per_bus = {}  # New dictionary to track feed-in power

    for result in result_jsons:
        bus_energies = result['bus_energies']
        day_ahead_prices = result['day_ahead_episode']

        for timestep, bus_energy in enumerate(bus_energies):
            for bus_id, bus_data in bus_energy.items():
                to_grid_energy = bus_data['to_grid']
                gain_for_timestep = to_grid_energy * day_ahead_prices[timestep] - 0.05 / 4

                # Accumulate total gain for each bus
                if bus_id not in total_feed_in_gain_per_bus:
                    total_feed_in_gain_per_bus[bus_id] = 0
                    total_feed_in_power_per_bus[bus_id] = 0
                total_feed_in_gain_per_bus[bus_id] += gain_for_timestep
                total_feed_in_power_per_bus[bus_id] += to_grid_energy / 4 # Accumul
    # Sum up the total feed-in gain made by all buses
    total_feed_in_gain = sum(total_feed_in_gain_per_bus.values())

    average_feed_in_gain_bus = total_feed_in_gain / len(total_feed_in_gain_per_bus) if total_feed_in_gain_per_bus else 0

    print(f'total_feed_in_gain_per_bus: {total_feed_in_gain_per_bus}')
    print(f'average_feed_in_gain_bus: {average_feed_in_gain_bus}')
    return total_feed_in_gain, average_feed_in_gain_bus, total_feed_in_gain_per_bus, total_feed_in_power_per_bus


def calculate_average_energy_cost(result_jsons):
    total_energy_cost_per_bus = {}
    total_energy_consumed_per_bus = {}

    for result in result_jsons:
        bus_energies = result['bus_energies']
        day_ahead_prices = result['day_ahead_episode']

        for timestep, bus_energy in enumerate(bus_energies):
            for bus_id, bus_data in bus_energy.items():
                from_grid_energy = bus_data['from_grid']
                # Calculate the cost for this timestep: (price * 1.03 + 0.16) for the energy consumed from the grid
                cost_for_timestep = from_grid_energy * (day_ahead_prices[timestep] * 1.03 + 0.16) / 4

                # Accumulate total cost and total energy consumed for each bus
                if bus_id not in total_energy_cost_per_bus:
                    total_energy_cost_per_bus[bus_id] = 0
                    total_energy_consumed_per_bus[bus_id] = 0

                total_energy_cost_per_bus[bus_id] += cost_for_timestep
                total_energy_consumed_per_bus[bus_id] += from_grid_energy / 4  # Adjust for 15-minute intervals

    # Sum up the total cost and total energy consumed by all buses
    total_energy_cost = sum(total_energy_cost_per_bus.values())
    total_energy_consumed = sum(total_energy_consumed_per_bus.values())

    # Calculate the average cost per bus
    average_energy_cost_per_bus = total_energy_cost / len(total_energy_cost_per_bus) if total_energy_cost_per_bus else 0

    # Calculate the price per kWh for each bus and the overall average
    price_per_kWh_per_bus = {bus_id: (cost / total_energy_consumed_per_bus[bus_id] if total_energy_consumed_per_bus[bus_id] > 0 else 0) for
                             bus_id, cost in total_energy_cost_per_bus.items()}
    average_price_per_kWh = total_energy_cost / total_energy_consumed if total_energy_consumed > 0 else 0

    print(f'total_energy_cost_per_bus: {total_energy_cost_per_bus}')
    print(f'average_energy_cost_per_bus: {average_energy_cost_per_bus}')
    print(f'price_per_kWh_per_bus: {price_per_kWh_per_bus}')
    print(f'average_price_per_kWh: {average_price_per_kWh}')

    return total_energy_cost, average_energy_cost_per_bus, price_per_kWh_per_bus, average_price_per_kWh, total_energy_consumed_per_bus, total_energy_cost_per_bus


def calculate_remaining_pv_percentage(result_jsons):
    total_remaining_pv_per_bus = {}
    total_pv_production_per_bus = {}
    remaining_pv_percentage_per_bus = {}

    for result_json in result_jsons:
        bus_energies = result_json['bus_energies']
        renewable = result_json['renewable']

        for timestep_data in bus_energies:
            for bus_id, energy_data in timestep_data.items():
                if 'pv_available' in energy_data:  # Check if PV data is available for the bus
                    pv_available = energy_data['pv_available']
                    if bus_id not in total_remaining_pv_per_bus:
                        total_remaining_pv_per_bus[bus_id] = 0
                    total_remaining_pv_per_bus[bus_id] += pv_available

        for bus_id, production_values in renewable.items():
            if bus_id not in total_pv_production_per_bus:
                total_pv_production_per_bus[bus_id] = 0
            total_pv_production_per_bus[bus_id] += sum(production_values)

    for bus_id in total_remaining_pv_per_bus:
        if bus_id in total_pv_production_per_bus and total_pv_production_per_bus[bus_id] > 0:
            remaining_pv_percentage_per_bus[bus_id] = (total_remaining_pv_per_bus[bus_id] / total_pv_production_per_bus[bus_id]) * 100
        else:
            remaining_pv_percentage_per_bus[
                bus_id] = 0  # Set to 0 if no PV production for this bus or bus doesn't exist in total_pv_production_per_bus

    total_remaining_pv = sum(total_remaining_pv_per_bus.values())
    total_pv_production = sum(total_pv_production_per_bus.values())
    if total_pv_production > 0:
        average_remaining_pv_percentage = (total_remaining_pv / total_pv_production) * 100
        print(f"Average remaining PV percentage over all buses: {average_remaining_pv_percentage:.2f}%")
    else:
        print("Total PV production is 0, cannot calculate average remaining PV percentage.")
        average_remaining_pv_percentage = None

    return remaining_pv_percentage_per_bus, average_remaining_pv_percentage

    # return remaining_pv_percentage_per_bus


def calculate_average_line_load(result_jsons):
    total_line_loads = []
    average_line_load_per_line = []
    total_lines = None
    exceed_100_count_per_line = []  # Count of exceedances per line
    timesteps_exceeding_100_count = 0  # Count of timesteps with any line exceeding 100%

    for result_json in result_jsons:
        line_loads = result_json['line_load']
        timestep_exceeded = False

        if total_lines is None:
            total_lines = len(line_loads[0])
            total_line_loads = [0] * total_lines
            exceed_100_count_per_line = [0] * total_lines  # Initialize count array

        for timestep_loads in line_loads:
            for i, load in enumerate(timestep_loads):
                total_line_loads[i] += load

                if load > 100:
                    exceed_100_count_per_line[i] += 1  # Increment count for this specific line
                    timestep_exceeded = True

            if timestep_exceeded:
                timesteps_exceeding_100_count += 1  # Increment once for this timestep

    # Calculate average load for each line
    total_timesteps = len(result_jsons) * len(result_jsons[0]['line_load'])
    for line_total_load in total_line_loads:
        average_line_load_per_line.append(line_total_load / total_timesteps)

    overall_average_line_load = sum(average_line_load_per_line) / total_lines if total_lines else 0

    return average_line_load_per_line, overall_average_line_load, exceed_100_count_per_line, timesteps_exceeding_100_count


# def calculate_net_grid_energy(result_jsons):
#     net_energy_from_grid_per_bus = {}
#
#     for result in result_jsons:
#         renewable_energies = result['renewable']  # PV generation per bus
#         load_energies = result['load']  # Load consumption per bus
#         ess_values = result['p_ess_filledevol']  # ESS interaction values per timestep for all buses
#         cs_values = result['p_cs_filledevol']  # CS interaction values per timestep for all buses
#
#         max_timesteps = 96  # Limiting to the first 96 timesteps
#
#         for bus_id in load_energies.keys():
#             net_energy_from_grid_per_bus[bus_id] = 0
#
#             for timestep in range(max_timesteps):
#                 load = load_energies[bus_id][timestep] if timestep < len(load_energies[bus_id]) else load_energies[bus_id][-1]
#
#                 pv_production = renewable_energies.get(bus_id, [0] * max_timesteps)[timestep] if timestep < len(renewable_energies.get(bus_id, [0] * max_timesteps)) else renewable_energies.get(bus_id, [0] * max_timesteps)[-1]
#
#                 ess_interaction = ess_values[timestep][int(bus_id)] if int(bus_id) < len(ess_values[timestep]) else ess_values[-1][int(bus_id)] if len(ess_values) > 0 and int(bus_id) < len(ess_values[-1]) else 0
#
#                 cs_interaction = cs_values[timestep][int(bus_id)] if int(bus_id) < len(cs_values[timestep]) else cs_values[-1][int(bus_id)] if len(cs_values) > 0 and int(bus_id) < len(cs_values[-1]) else 0
#
#                 net_load = load - pv_production - ess_interaction - cs_interaction
#                 net_load_from_grid = max(net_load, 0)
#                 net_energy_from_grid_per_bus[bus_id] += net_load_from_grid / 4  # Convert power to energy for the 15-minute interval
#
#     return net_energy_from_grid_per_bus

def calculate_grid_energy_requirement(result_jsons):
    grid_energy_requirement_per_bus = {}

    for result in result_jsons:
        renewable_energies = result['renewable']  # PV generation per bus
        load_energies = result['load']  # Load consumption per bus
        ess_values = result['p_ess_filledevol']  # ESS interaction values per timestep for all buses
        cs_values = result['p_cs_filledevol']  # CS interaction values per timestep for all buses

        max_timesteps = 96  # Limiting to the first 96 timesteps

        for bus_id in load_energies.keys():
            bus_id_int = int(bus_id)  # Convert bus_id to integer to match the list index
            grid_energy_requirement_per_bus[bus_id] = 0

            # Ensure bus_id_int is within the range of ess_values and cs_values lists
            if bus_id_int < len(ess_values) and bus_id_int < len(cs_values):
                for timestep in range(max_timesteps):
                    # Check if timestep is within the sublist's length
                    if timestep < len(ess_values[bus_id_int]) and timestep < len(cs_values[bus_id_int]):
                        local_source_contribution = renewable_energies.get(bus_id, [0] * max_timesteps)[timestep] + \
                                                    ess_values[bus_id_int][timestep] + \
                                                    cs_values[bus_id_int][timestep]

                        # Determine the shortfall by subtracting the local contribution from the load
                        shortfall = load_energies[bus_id][timestep] - local_source_contribution

                        # If there's a shortfall (positive value), it needs to be covered by the grid
                        if shortfall > 0:
                            grid_energy_requirement_per_bus[bus_id] += shortfall

    # Print the grid energy requirement per bus
    for bus_id, requirement in grid_energy_requirement_per_bus.items():
        print(f"Bus ID {bus_id}: Grid energy requirement = {requirement} kWh")

    return grid_energy_requirement_per_bus
def calculate_average_kWh_price_cs_from_grid(result_jsons):
    total_cost_per_bus = {}
    total_energy_per_bus = {}

    for result in result_jsons:
        bus_energies = result['bus_energies']  # List of dictionaries for each timestep
        day_ahead_prices = result['day_ahead_episode']  # List of prices for each timestep
        power_cs_day = result['power_cs']  # Power values for all charging stations, assuming it's a list of lists

        for timestep, bus_energy in enumerate(bus_energies):
            if timestep >= len(power_cs_day):  # Check if the timestep is within the range of power_cs_day
                continue

            for bus_id, bus_data in bus_energy.items():
                if not 'cs_from_grid' in bus_data:  # Check if cs data is available for this bus at this timestep
                    continue

                cs_from_grid = bus_data['cs_from_grid']  # Power from grid for this bus at this timestep
                bus_index = int(bus_id)  # Adjust this if necessary to match bus_id with the index in power_cs_day

                if bus_index < len(power_cs_day[timestep]):  # Check if bus_index is within the range for this timestep
                    cs_total_for_bus = power_cs_day[timestep][bus_index]  # Total power used by this bus
                else:
                    cs_total_for_bus = 0  # If no data, assume 0

                free_energy_for_bus = max(0, cs_total_for_bus - cs_from_grid)  # Free energy used by this bus
                total_energy_for_bus = cs_from_grid + free_energy_for_bus  # Total energy used by this bus

                energy_cost = day_ahead_prices[timestep] * cs_from_grid / 4  # Cost for this timestep, considering 15-minute interval

                # Accumulate total energy and cost for each bus
                if bus_id not in total_energy_per_bus:
                    total_energy_per_bus[bus_id] = 0
                    total_cost_per_bus[bus_id] = 0

                total_energy_per_bus[bus_id] += total_energy_for_bus / 4  # Convert power to energy for the 15-minute interval
                total_cost_per_bus[bus_id] += energy_cost

    # Calculate average price per kWh for each bus
    average_price_per_bus = {bus_id: total_cost / total_energy if total_energy > 0 else 0
                             for bus_id, total_cost, total_energy in
                             zip(total_cost_per_bus.keys(), total_cost_per_bus.values(), total_energy_per_bus.values())}

    return average_price_per_bus


def calculate_net_energy_for_ess_cs_load_pv(result_jsons):
    net_energy_ess_per_bus = {}  # Net energy for each bus from ESS
    net_energy_cs_per_bus = {}  # Net energy for each bus from CS
    net_energy_load_per_bus = {}  # Net energy for each bus from Load
    total_pv_production_per_bus = {}  # Total PV energy production per bus

    for result in result_jsons:
        power_ess = result.get('power_ess', [])
        power_cs = result.get('power_cs', [])
        load = result.get('load', {})
        renewable = result.get('renewable', {})

        for timestep, ess_powers in enumerate(power_ess):
            for bus_id, power in enumerate(ess_powers):
                if bus_id not in net_energy_ess_per_bus:
                    net_energy_ess_per_bus[bus_id] = 0
                net_energy_ess_per_bus[bus_id] += power / 4

        for timestep, cs_powers in enumerate(power_cs):
            for bus_id, power in enumerate(cs_powers):
                if bus_id not in net_energy_cs_per_bus:
                    net_energy_cs_per_bus[bus_id] = 0
                net_energy_cs_per_bus[bus_id] += power / 4

        for bus_id, loads in load.items():
            bus_id_int = int(bus_id)  # Convert bus_id to integer
            if bus_id_int not in net_energy_load_per_bus:
                net_energy_load_per_bus[bus_id_int] = 0
            net_energy_load_per_bus[bus_id_int] += sum(loads) / 4  # Convert to kWh

        for bus_id, production_values in renewable.items():
            bus_id_int = int(bus_id)  # Convert bus_id to integer
            if bus_id_int not in total_pv_production_per_bus:
                total_pv_production_per_bus[bus_id_int] = 0
            total_pv_production_per_bus[bus_id_int] += sum(production_values) / 4  # Convert to kWh

    return net_energy_ess_per_bus, net_energy_cs_per_bus, net_energy_load_per_bus, total_pv_production_per_bus



# def calculate_average_kWh_load(results_json):
#
#     pass#
#     return average_kWh_load_per_bus, average_kWh_load


total_feed_in_gain, average_feed_in_gain_bus, total_feed_in_gain_per_bus, total_feed_in_power_per_bus = calculate_total_feed_in_gain(
    result_jsons=result_jsons)

# average_line_load_per_line, overall_average_line_load, exceed_100_count_per_line, timesteps_exceeding_100_count = calculate_average_line_load(
#     result_jsons=result_jsons)
# Calculate the average price per bus
average_price_cs_per_bus = calculate_average_kWh_price_cs_from_grid(result_jsons)

total_energy_cost, average_energy_cost_per_bus, price_per_kWh_per_bus, average_price_per_kWh, total_energy_consumed_per_bus, total_energy_cost_per_bus = calculate_average_energy_cost(
    result_jsons)
print(average_price_cs_per_bus)

# Calculate average SOC
car_soc_values_dep, overall_average_soc_dep = calculate_average_soc_on_leave(result_jsons)

remaining_pv_percentage_per_bus, average_remaining_pv_percentage = calculate_remaining_pv_percentage(result_jsons)
#
# print(average_soc)
calculate_grid_energy_requirement(result_jsons)


net_energy_ess_per_bus, net_energy_cs_per_bus, net_energy_load_per_bus, total_pv_production_per_bus = calculate_net_energy_for_ess_cs_load_pv(result_jsons)
print("Net Energy ESS per Bus:", net_energy_ess_per_bus)
print("Net Energy CS per Bus:", net_energy_cs_per_bus)
print("Net Energy Load per Bus:", net_energy_load_per_bus)
print("Total PV Production per Bus:", total_pv_production_per_bus)

# average_kWh_load_per_bus, average_kWh_load = calculate_average_kWh_load(result_jsons)

net_load_from_grid_per_bus = calculate_net_grid_energy(result_jsons)



eval_metrics = {
    'total_feed_in_gain': total_feed_in_gain,
    'average_feed_in_gain_per_bus': average_feed_in_gain_bus,
    'total_feed_in_gain_per_bus': total_feed_in_gain_per_bus,
    'total_feed_in_power_per_bus': total_feed_in_power_per_bus,
    'average_price_per_bus': average_price_cs_per_bus,
    'overall_average_soc_departure': overall_average_soc_dep,
    'car_soc_values_dep': car_soc_values_dep,
    'average_remaining_pv_percentage': average_remaining_pv_percentage,
    'remaining_pv_percentage_per_bus': remaining_pv_percentage_per_bus,
    # 'average_line_load_per_line': average_line_load_per_line,
    # 'overall_average_line_load': overall_average_line_load,
    # 'exceed_100_count_per_line': exceed_100_count_per_line,
    # 'timesteps_exceeding_100_count': timesteps_exceeding_100_count,
    'total_energy_cost': total_energy_cost,
    'average_energy_cost_per_bus': average_energy_cost_per_bus,
    'total_energy_consumed_per_bus': total_energy_consumed_per_bus,
    'total_energy_cost_per_bus': total_energy_cost_per_bus,
    'price_per_kWh_per_bus': price_per_kWh_per_bus,
    'average_price_per_kWh': average_price_per_kWh,
    "Net Energy ESS per Bus:": net_energy_ess_per_bus,
    "Net Energy CS per Bus:": net_energy_cs_per_bus,
    "Net Energy Load per Bus:": net_energy_load_per_bus,
    "Total PV Production per Bus:": total_pv_production_per_bus,
    'net_load_from_grid_per_bus': net_load_from_grid_per_bus

}

# Specify the directory and file name
file_path = f'Results/{model_type}_{session_prefix}{session_id}/_eval_metrics.json'

# Ensure the directory exists
# Write the results to a JSON file
with open(file_path, 'w') as file:
    json.dump(eval_metrics, file, indent=4)

print(f"Saved evaluation metrics to '{file_path}'")
