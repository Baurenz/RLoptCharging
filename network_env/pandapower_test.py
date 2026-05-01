import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import pandapower as pp
import pandas as pd
from pandapower.timeseries import DFData, OutputWriter
from pandapower.control import ConstControl
from pandapower.timeseries.run_time_series import run_timeseries

import logging
# logging.basicConfig(level=logging.DEBUG)

# Load the network from the JSON file
net = pp.from_json('../data/scenarios/2-Bus_2-PV_2-Load_2-V2G_externalgrid_bus1_no_data.json')
data_path = "../data/"

# Load consumption and production profiles
df_pv1 = pd.read_csv(data_path + "pv1.csv", index_col=0)
df_pv2 = pd.read_csv(data_path + "pv2.csv", index_col=0)
df_consumption1 = pd.read_csv(data_path + "consumption1.csv", index_col=0)
df_consumption2 = pd.read_csv(data_path + "consumption2.csv", index_col=0)
df_v2g1 = pd.read_csv(data_path + "v2g1.csv", index_col=0)
df_v2g2 = pd.read_csv(data_path + "v2g2.csv", index_col=0)

# Create data sources from the loaded DataFrames
ds_pv1 = DFData(df_pv1)
ds_pv2 = DFData(df_pv2)
ds_consumption1 = DFData(df_consumption1)
ds_consumption2 = DFData(df_consumption2)
ds_v2g1 = DFData(df_v2g1)
ds_v2g2 = DFData(df_v2g2)

# Define what to output

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'./results/{timestamp}'

ow = OutputWriter(net, output_path=output_path, output_file_type=".csv")

# Find the element indices by their names
pv1_idx = net.sgen[net.sgen.name == "PV1"].index[0]
pv2_idx = net.sgen[net.sgen.name == "PV2"].index[0]
load1_idx = net.load[net.load.name == "Load1"].index[0]
load2_idx = net.load[net.load.name == "Load2"].index[0]
battery1_idx = net.storage[net.storage.name == "Battery1"].index[0]
battery2_idx = net.storage[net.storage.name == "Battery2"].index[0]

ConstControl(net, element='sgen', element_index=pv1_idx, variable='p_mw', data_source=ds_pv1, profile_name="PV1")
ConstControl(net, element='sgen', element_index=pv2_idx, variable='p_mw', data_source=ds_pv2, profile_name="PV2")
ConstControl(net, element='load', element_index=load1_idx, variable='p_mw', data_source=ds_consumption1,
             profile_name="Load1")
ConstControl(net, element='load', element_index=load2_idx, variable='p_mw', data_source=ds_consumption2,
             profile_name="Load2")
ConstControl(net, element='storage', element_index=battery1_idx, variable='p_mw', data_source=ds_v2g1,
             profile_name="V2G1")
ConstControl(net, element='storage', element_index=battery2_idx, variable='p_mw', data_source=ds_v2g2,
             profile_name="V2G2")

# Define what to output
ow.log_variable('res_load', 'p_mw')
ow.log_variable('res_sgen', 'p_mw')
ow.log_variable('res_storage', 'p_mw')
ow.log_variable('res_ext_grid', 'p_mw')
ow.log_variable('res_line', 'p_from_mw')

# Run the time series simulation for the steps in the profiles
time_steps = df_pv1.index.tolist()


# Create a dictionary of the DataFrames
dataframes = {
    'df_pv1': df_pv1,
    'df_pv2': df_pv2,
    'df_consumption1': df_consumption1,
    'df_consumption2': df_consumption2,
    'df_v2g1': df_v2g1,
    'df_v2g2': df_v2g2,
}

# Call the function to check the index existence
# check_index_existence(dataframes)

line_idx = net.line.index[0]

# net.line['max_loading_percent'] = 5
net.line.at[line_idx, 'max_loading_percent'] = 4

# result_dict = pp.diagnostic(net, report_style='detailed')

run_timeseries(net, time_steps=time_steps, verbose=True)
# Call the visualization function


def visualize_timeseries_results(run_id=timestamp):
    # Define the path to the results
    results_path = Path(f"./results/{run_id}/")

    # Load the logged data
    res_load = pd.read_csv(results_path / "res_load" / "p_mw.csv", delimiter=";", index_col=0)
    res_sgen = pd.read_csv(results_path / "res_sgen" / "p_mw.csv", delimiter=";", index_col=0)
    res_storage = pd.read_csv(results_path / "res_storage" / "p_mw.csv", delimiter=";", index_col=0)
    res_ext_grid = pd.read_csv(results_path / "res_ext_grid" / "p_mw.csv", delimiter=";", index_col=0)
    res_line = pd.read_csv(results_path / "res_line" / "p_from_mw.csv", delimiter=";", index_col=0)
    line_loading = pd.read_csv(results_path / "res_line" / "loading_percent.csv", delimiter=";", index_col=0)

    # Set up the plots
    fig, axes = plt.subplots(5, 1, figsize=(14, 24))  # Increased the number of subplots

    # Node 1
    axes[0].plot(res_load["0"], label="Consumption (Node 1)")
    axes[0].plot(res_sgen["0"], label="Production (Node 1)", linestyle="--")
    axes[0].set_title("Production and Consumption at Node 1")
    axes[0].set_ylabel("Power (MW)")
    axes[0].legend()

    # Node 2
    axes[1].plot(res_load["1"], label="Consumption (Node 2)")
    axes[1].plot(res_sgen["1"], label="Production (Node 2)", linestyle="--")
    axes[1].set_title("Production and Consumption at Node 2")
    axes[1].set_ylabel("Power (MW)")
    axes[1].legend()

    # External Grid
    axes[2].plot(res_ext_grid["0"], label="Power Flow to External Grid")  # Assuming "0" is the index here
    axes[2].set_title("Power Flow to External Grid")
    axes[2].set_ylabel("Power (MW)")

    # Power Flow of Storages
    axes[3].plot(res_storage["0"], label="Power Flow of Storage 1")  # Assuming "0" for Storage 1
    axes[3].plot(res_storage["1"], label="Power Flow of Storage 2")  # Assuming "1" for Storage 2
    axes[3].set_title("Power Flow of Storages")
    axes[3].set_ylabel("Power (MW)")
    axes[3].legend()

    # Line Flow & Loading Percentage
    line_name = "0"  # Replace with your line's index/name, assuming "0" based on your pattern
    axes[4].plot(res_line[line_name], label="Power Flow through Line", color='b')
    axes[4].set_title("Power Flow through Line and its Loading Percentage")
    axes[4].set_ylabel("Power (MW)", color='b')
    axes[4].tick_params(axis='y', labelcolor='b')

    ax2 = axes[4].twinx()  # instantiate a second y-axis
    ax2.plot(line_loading[line_name], label="Loading Percentage of Line", color='r')
    ax2.set_ylabel("Loading Percentage (%)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # Choose the number of bins you prefer
    # Adjust layout and show
    plt.tight_layout()
    plt.show()


visualize_timeseries_results(timestamp)

