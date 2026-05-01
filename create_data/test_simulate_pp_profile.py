import pandapower as pp
from pandapower.timeseries import DFData, OutputWriter
from pandapower.control import ConstControl
from pandapower.timeseries.run_time_series import run_timeseries

# 1. Read saved pandapower network from JSON
net = pp.from_json('../data/scenarios/Dynamic_Network_2023-07-21_no_data.json')

# 2. Create a DataSource object and load required profiles
# (Assuming you've stored the profiles in CSV files)
profiles_df = {}

for bus, comps in components.items():
    for comp_key, comp_value in comps.items():
        profile_name = f"{bus}_{comp_key}"
        profile_file_name = f"dynamic_profiles/{profile_name}.csv"
        profiles_df[profile_name] = pd.read_csv(f'../data/{profile_file_name}', index_col=0)

# Convert all profiles to a single DataFrame
profiles_df = pd.concat(profiles_df, axis=1)
ds = DFData(profiles_df)

# 3. Define OutputWriter to store results and ConstControl to apply profiles
ow = OutputWriter(net, time_steps=range(len(time_index)), output_path="./results", output_file_type=".json")

# Map profiles to respective components
for bus, comps in components.items():
    for comp_key, comp_value in comps.items():
        profile_name = f"{bus}_{comp_key}"
        element_index = net[comp_key][net[comp_key]["name"] == comp_value].index[0]

        if comp_key == "pv":
            ConstControl(net, element="sgen", variable="p_mw", element_index=element_index,
                         data_source=ds, profile_name=profile_name)
        elif comp_key == "load":
            ConstControl(net, element="load", variable="p_mw", element_index=element_index,
                         data_source=ds, profile_name=profile_name)
        # Add other component controls similarly...

ow.log_variable('res_bus', 'vm_pu')
# Add other variables as needed

# 4. Run the time series simulation
run_timeseries(net, time_steps=range(len(time_index)))

# Now, you should have results stored in the "./results" directory.
