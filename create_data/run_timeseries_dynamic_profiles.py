import json
from datetime import date
import time

import numpy as np
import pandapower as pp
import pandas as pd

import matplotlib

from matplotlib import pyplot as plt

from pandapower.control import ConstControl
from pandapower.timeseries import DFData, OutputWriter, run_timeseries


# matplotlib.use('TkAgg')
# Loading Functions and Setup
def load_network():
    today = str(date.today())
    file_name = f'../data/scenarios/Dynamic_Network_{today}_no_data.json'
    return pp.from_json(file_name)


def load_profiles():
    profiles = {}
    with open("profile_json/dynamic_profile_6bus.json", "r") as file:
        data = json.load(file)
        components = data["components"]
    for bus, comps in components.items():
        for comp_key, _ in comps.items():
            profile_name = f"{bus}_{comp_key}"
            profile_file_name = f'../data/dynamic_profiles/{profile_name}.csv'
            profiles[profile_name] = pd.read_csv(profile_file_name, index_col=0, parse_dates=True)
    return profiles


# Time Series Functions
def setup_time_series(net, profiles):
    ds_dict = {profile_name: DFData(profile_df) for profile_name, profile_df in profiles.items()}
    for profile_name in profiles:
        bus, comp_type = profile_name.split("_")
        bus_index = net.bus[net.bus.name == bus].index[0]

        if comp_type == "pv":
            sgen_idx = net.sgen[(net.sgen.bus == bus_index) & (net.sgen.name.str.startswith(comp_type))].index
            for idx in sgen_idx:
                ConstControl(net, element='sgen', element_index=idx, variable='p_mw',
                             data_source=ds_dict[profile_name], profile_name=profile_name, in_service=True)

        elif comp_type == "load":
            load_idx = net.load[(net.load.bus == bus_index) & (net.load.name.str.startswith(comp_type))].index
            for idx in load_idx:
                ConstControl(net, element='load', element_index=idx, variable='p_mw',
                             data_source=ds_dict[profile_name], profile_name=profile_name, in_service=True)


# Time Series Execution and Result Gathering
def execute_time_series(net, profiles):
    ow = OutputWriter(net, time_steps=profiles[next(iter(profiles))].index, output_path="./results",
                      output_file_type=".json")

    ow.log_variable('res_sgen', 'p_mw')
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_ext_grid', 'p_mw')
    ow.log_variable('res_line', 'loading_percent')

    ow.log_variable('res_line', 'p_from_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'p_mw')  # Log active power at each bus

    start_time = time.time()
    run_timeseries(net, time_steps=profiles[next(iter(profiles))].index)
    end_time = time.time()

    print(f"Simulation executed in {end_time - start_time} seconds")

    bus_pvs = ow.output["res_sgen.p_mw"].transpose().to_dict()
    bus_loads = ow.output["res_load.p_mw"].transpose().to_dict()

    ext_grid_p_mw = ow.output["res_ext_grid.p_mw"].values.tolist()

    # Using OutputWriter to extract the results for each time step
    line_percentage = ow.output["res_line.loading_percent"].values.tolist()
    line_p = None

    bus_p_mw = ow.output["res_bus.p_mw"].transpose().to_dict()

    return line_p, line_percentage, ext_grid_p_mw, bus_pvs, bus_loads, bus_p_mw


def plot_line_percentages(line_percentage, profiles):
    """
    Plots the line loading percentages of all lines in a single plot.

    Parameters:
    - line_percentage: List of loading percentages for each time step for all lines.
    - profiles: Dictionary of loaded profile DataFrames.
    """
    # Extracting timestamps from the first profile DataFrame
    timestamps = profiles[next(iter(profiles))].index

    plt.figure(figsize=(10, 6))

    # We transpose line_percentage to iterate through lines instead of timestamps
    for idx, line_load in enumerate(zip(*line_percentage)):
        plt.plot(timestamps, line_load, label=f'Line {idx + 1}')

    plt.xlabel('Time')
    plt.ylabel('Loading Percentage (%)')
    plt.title('Line Loading Percentages Over Time')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plot_ext_grid_power_flow(ext_grid_p_mw, profiles):
    timestamps = profiles[next(iter(profiles))].index

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, ext_grid_p_mw, label='Ext Grid Power Flow')
    plt.xlabel('Time')
    plt.ylabel('P_MW')
    plt.title('Power Flow to External Grid')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plot_bus_profiles(bus_pvs, bus_loads, bus_p_mw):
    """
    Plots the PV production, load, and active power for each bus in separate subplots.

    Parameters:
    - bus_pvs: Dictionary containing PV values for each timestamp. Each entry is a dictionary of bus values.
    - bus_loads: Similar structure to bus_pvs but for load values.
    - bus_p_mw: Similar structure to bus_pvs but for active power values.
    """
    # Extracting timestamps from bus_pvs
    timestamps = list(bus_pvs.keys())

    # Assuming all dictionaries have the same set of buses, extracting buses from the first timestamp
    buses = list(bus_pvs[timestamps[0]].keys())

    fig, axes = plt.subplots(len(buses), 1, figsize=(10, 6 * len(buses)))

    for idx, bus in enumerate(buses):
        # Change the sign of PV values
        pv_values = [-bus_pvs[ts][bus] for ts in timestamps]
        load_values = [bus_loads[ts].get(bus, 0) for ts in timestamps]
        power_values = [bus_p_mw[ts].get(bus, 0) for ts in timestamps]

        # Plotting PV values with dashed line
        axes[idx].plot(timestamps, pv_values, label=f'{bus} PV', linestyle='--', color='blue')

        # Plotting Load values with dashed line
        axes[idx].plot(timestamps, load_values, label=f'{bus} Load', linestyle='--', color='red')

        # Plotting Active Power values with thick black line
        axes[idx].plot(timestamps, power_values, label=f'{bus} P_MW', linestyle='-', color='black', linewidth=2)

        axes[idx].set_xlabel('Time')
        axes[idx].set_ylabel('P_MW')
        axes[idx].set_title(f'PV, Load, and P_MW for Bus: {bus}')
        axes[idx].legend(loc='upper right')
        axes[idx].grid(True)

    plt.tight_layout()
    plt.show()


def plot_system_balance(bus_pvs, bus_loads, ext_grid_p_mw):
    """
    Plots the aggregated system PV production, load, and compares it with the grid import/export.

    Parameters:
    - bus_pvs: Dictionary containing PV values for each timestamp. Each entry is a dictionary of bus values.
    - bus_loads: Similar structure to bus_pvs but for load values.
    - ext_grid_p_mw: List containing grid power flow values over time.
    """
    # Extracting timestamps from bus_pvs
    timestamps = list(bus_pvs.keys())

    # Calculate total PV, Load, and P_MW across all buses for each timestamp
    total_pv = [-sum(bus_pvs[ts].values()) for ts in timestamps]
    total_load = [sum(bus_loads[ts].values()) for ts in timestamps]
    net_balance = [pv + load for pv, load in zip(total_pv, total_load)]

    plt.figure(figsize=(10, 6))

    # Plotting aggregated values
    plt.plot(timestamps, total_pv, label='Total PV', linestyle='--', color='blue')
    plt.plot(timestamps, total_load, label='Total Load', linestyle='--', color='red')
    plt.plot(timestamps, net_balance, label='Net Balance (PV - Load)', linestyle='-', color='black', linewidth=2)
    plt.plot(timestamps, ext_grid_p_mw, label='Grid Import/Export', linestyle='-.', color='green', linewidth=1.5)

    plt.xlabel('Time')
    plt.ylabel('P_MW')
    plt.title('System-wide Balance Over Time')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main Execution
net = load_network()
profiles = load_profiles()
setup_time_series(net, profiles)
line_p, line_percentage, ext_grid_p_mw, bus_pvs, bus_loads, bus_p_mw = execute_time_series(net, profiles)

plot_line_percentages(line_percentage, profiles)
plot_ext_grid_power_flow(ext_grid_p_mw, profiles)
plot_bus_profiles(bus_pvs, bus_loads, bus_p_mw)
# Adjust main execution section to include the new function:
plot_system_balance(bus_pvs, bus_loads, ext_grid_p_mw)
