import pandapower as pp
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.control import ConstControl
from pandapower.timeseries.run_time_series import run_timeseries
import pandas as pd

# Read the network from the JSON file
net = pp.from_json("../data/scenarios/test.json")
print(net.load)

# Load consumption and production profiles
df_consumption1 = pd.read_csv("../data/consumption1.csv", index_col=0)
df_consumption2 = pd.read_csv("../data/consumption2.csv", index_col=0)
df_gen1 = pd.read_csv("../data/gen1.csv", index_col=0)
df_gen2 = pd.read_csv("../data/gen2.csv", index_col=0)
df_pv1 = pd.read_csv("../data/pv1.csv", index_col=0)
df_pv2 = pd.read_csv("../data/pv2.csv", index_col=0)

# Convert the index to datetime
dfs = [df_consumption1, df_consumption2, df_gen1, df_gen2, df_pv1, df_pv2]
for df in dfs:
    df.index = pd.to_datetime(df.index)

# Create data sources from the loaded DataFrames
ds_consumption1 = DFData(df_consumption1)
ds_consumption2 = DFData(df_consumption2)
ds_gen1 = DFData(df_gen1)
ds_gen2 = DFData(df_gen2)
ds_pv1 = DFData(df_pv1)
ds_pv2 = DFData(df_pv2)

# Initialize the data sources and connect them to the controllers
ConstControl(net, element='load', element_index=0, variable='p_mw', data_source=ds_consumption1)
ConstControl(net, element='load', element_index=1, variable='p_mw', data_source=ds_consumption2)
ConstControl(net, element='gen', element_index=0, variable='p_mw', data_source=ds_gen1)
ConstControl(net, element='gen', element_index=1, variable='p_mw', data_source=ds_gen2)
ConstControl(net, element='sgen', element_index=0, variable='p_mw', data_source=ds_pv1)
ConstControl(net, element='sgen', element_index=1, variable='p_mw', data_source=ds_pv2)

print(type(net["load"]))
print(net["load"])


time_steps = df_pv1.index.tolist()  # Assuming all CSV files have the same index

print(time_steps)
run_timeseries(net, time_steps= time_steps)
