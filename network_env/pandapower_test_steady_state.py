import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time

import pandapower as pp
import pandas as pd
from pandapower.timeseries import DFData, OutputWriter
from pandapower.control import ConstControl
from pandapower.timeseries.run_time_series import run_timeseries

# Load the network from the JSON file
net = pp.from_json('../data/scenarios/2-Bus_2-PV_2-Load_2-V2G_externalgrid_bus1_no_data.json')
data_path = "../data/"

# Load consumption and production profiles and convert their indices to time only
data_dict = {}
for data_name in ["pv1", "pv2", "consumption1", "consumption2", "v2g1", "v2g2"]:
    df = pd.read_csv(f"{data_path}{data_name}.csv", index_col=0)
    df.index = pd.to_datetime(df.index).time
    data_dict[f"ds_{data_name}"] = DFData(df)

# Create index dictionary for network elements
idx_dict = {
    'pv1_idx': net.sgen[net.sgen.name == "PV1"].index[0],
    'pv2_idx': net.sgen[net.sgen.name == "PV2"].index[0],
    'load1_idx': net.load[net.load.name == "Load1"].index[0],
    'load2_idx': net.load[net.load.name == "Load2"].index[0],
    'battery1_idx': net.storage[net.storage.name == "Battery1"].index[0],
    'battery2_idx': net.storage[net.storage.name == "Battery2"].index[0]
}

# Define what to output
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'./results/{timestamp}'
ow = OutputWriter(net, output_path=output_path, output_file_type=".csv")


def run_single_timestep_pf(net, time_step, data_dict, idx_dict):
    start_time = time.time()

    pv1_value = data_dict['ds_pv1'].df.loc[time_step, "PV1"]
    pv2_value = data_dict['ds_pv2'].df.loc[time_step, "PV2"]
    consumption1_value = data_dict['ds_consumption1'].df.loc[time_step, "Load1"]
    consumption2_value = data_dict['ds_consumption2'].df.loc[time_step, "Load2"]
    v2g1_value = data_dict['ds_v2g1'].df.loc[time_step, "V2G1"]
    v2g2_value = data_dict['ds_v2g2'].df.loc[time_step, "V2G2"]

    net.sgen.at[idx_dict['pv1_idx'], "p_mw"] = pv1_value
    net.sgen.at[idx_dict['pv2_idx'], "p_mw"] = pv2_value
    net.load.at[idx_dict['load1_idx'], "p_mw"] = consumption1_value
    net.load.at[idx_dict['load2_idx'], "p_mw"] = consumption2_value
    net.storage.at[idx_dict['battery1_idx'], "p_mw"] = v2g1_value
    net.storage.at[idx_dict['battery2_idx'], "p_mw"] = v2g2_value

    pp.runpp(net)

    # Record the end time
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken for this timestep: {elapsed_time:.6f} seconds")

    print(f"Results for time_step: {time_step}")
    print("===================================")
    print("Load Results")
    print("------------")
    print(net.res_load)
    print("\nSGEN Results")
    print("------------")
    print(net.res_sgen)
    print("\nStorage Results")
    print("---------------")
    print(net.res_storage)
    print("\nExternal Grid Results")
    print("----------------------")
    print(net.res_ext_grid)
    print("\nLine Loading Results")
    print("---------------------")
    print(net.res_line)
    print("===================================")


# Time step for which the power flow should be calculated
time_step_to_run = '00:00:00'
time_step_to_run = datetime.datetime.strptime(time_step_to_run, '%H:%M:%S').time()

# Run the power flow calculation for the specific time step

for i in range(5):
    print(f"Run {i+1}")
    run_single_timestep_pf(net, time_step_to_run, data_dict, idx_dict)
