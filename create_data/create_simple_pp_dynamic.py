# TODO: implement some logic which creates a matching pandapower network to for
#  dynamic_profile_6bus.json/create_profiles_dynamic.py
from datetime import datetime, date
import json
import pandapower as pp
import os

today = str(date.today())

profile_json = "profile_json/simple_CS_noLoad_2bus.json"
base_name = os.path.splitext(os.path.basename(profile_json))[0]


# Load the JSON
with open(profile_json, "r") as file:
    data = json.load(file)

buses = data["buses"]
connections = data["connections"]
components = data["components"]
ext_grid_bus_name = data.get("ext_grid_bus", "bus0")  # Defaults to 'bus0' if not specified in the JSON

# Create an empty network
net = pp.create_empty_network()

# Step 1: Create buses
bus_names = [f"bus{i}" for i in range(buses)]
bus_ids = [pp.create_bus(net, vn_kv=0.4, name=bus_name) for bus_name in bus_names]

# Check if the specified ext_grid_bus is valid and add an external grid connection
if ext_grid_bus_name in bus_names:
    pp.create_ext_grid(net, bus=bus_ids[bus_names.index(ext_grid_bus_name)], vm_pu=1.02)
else:
    raise ValueError(f"'{ext_grid_bus_name}' is not a valid bus name in the provided JSON.")

# Step 2: Create connections (lines) based on the matrix
for i, row in enumerate(connections):
    for j, connection in enumerate(row):
        if j > i and connection:  # Only create connections for one half of the matrix
            pp.create_line(net, from_bus=bus_ids[i], to_bus=bus_ids[j], length_km=0.02, std_type="15-AL1/3-ST1A 0.4")

## possibility to create llines from custom parameters, only for testing purpuse
#
# high_resistance = 0.0001  # High resistance in ohms per kilometer
# high_reactance = 0.5  # Adjust reactance if needed
# high_capacity = 100  # 100 kA rating for each line to handle more current without overloading
#

# for i, row in enumerate(connections):
#     for j, connection in enumerate(row):
#         if j > i and connection:
#             pp.create_line_from_parameters(
#                 net,
#                 from_bus=bus_ids[i],
#                 to_bus=bus_ids[j],
#                 length_km=100.0,
#                 r_ohm_per_km=high_resistance,
#                 x_ohm_per_km=high_reactance,
#                 c_nf_per_km=1.0,
#                 max_i_ka=high_capacity
#             )

# Step 3: Add components to each bus
for bus_name, comps in components.items():
    # Get bus id
    bus_idx = bus_ids[int(bus_name.replace("bus", ""))]

    # Create PV if exists in components
    if "pv" in comps:
        name = comps["pv"]
        pp.create_sgen(net, bus=bus_idx, p_mw=0.0, q_mvar=0.0, name=f"pv{bus_name}")

    # # Create Battery if exists in components
    # if "battery_capacity" in comps:
    #     battery_capacity = comps["battery_capacity"]
    #     name = f"Battery_{bus_name}"
    #     pp.create_storage(net, bus=bus_idx, p_mw=0.01, q_mvar=0, sn_mva=0, soc_percent=100.0,
    #                       max_e_mwh=battery_capacity, name=name)

    # Create Load if exists in components
    if "load" in comps:
        load_name = comps["load"]
        pp.create_load(net, bus=bus_idx, p_mw=0.0, q_mvar=0.0, name=f"load{bus_name}")

    if "ess" in comps:
        storage_data = comps["ess"]
        pp.create_storage(
            net,
            bus=bus_idx,
            p_mw=storage_data.get("p_mw", 0.00),  # Default value if not provided
            # q_mvar=storage_data.get("q_mvar", 0.0),  # Default value if not provided
            max_e_mwh=storage_data["max_e_mwh"],
            soc_percent=storage_data.get("soc_percent", 100.0),  # Default value if not provided
            name=f"ess{bus_name}"
        )

    # serves as representation for the battery of my car
    if "cs" in comps:
        storage_data = comps["cs"]
        pp.create_storage(
            net,
            bus=bus_idx,
            p_mw=storage_data.get("p_mw", 0.0),  # Default value if not provided
            # q_mvar=storage_data.get("q_mvar", 0.0),  # Default value if not provided
            max_e_mwh=storage_data["max_e_mwh"],    # TODO: capacity of car battery is part of the car not the network. no need to define it here
            soc_percent=storage_data.get("soc", 100.0),  # Default value if not provided
            name=f"cs{bus_name}"
        )

# Save the network to JSON
num_buses = len(bus_names)

# Create the file name with the number of buses
file_name = f'../data/scenarios/{base_name}_no_data.json'

# Save the network to JSON
pp.to_json(net, filename=file_name)
