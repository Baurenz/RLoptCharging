# TODO: implement some logic which creates a matching pandapower network to for
#  dynamic_profile_6bus.json/create_profiles_dynamic.py
from datetime import datetime, date
import json
import pandapower as pp
import os

# Create new standard type
new_std_type = {"r_ohm_per_km": 0.2,  # Replace with your desired resistance value
                "x_ohm_per_km": 0.1,  # Example reactance value
                "c_nf_per_km": 0,  # Example capacitance value
                "max_i_ka": 0.05,  # Example maximum current value
                "type": "ol"}  # Overhead line type



def create_pp_network(profile_name):
    today = str(date.today())

    profile_json_path = f"data/profile_json/{profile_name}.json"

    base_name = os.path.splitext(os.path.basename(profile_json_path))[0]

    # Load the JSON
    with open(profile_json_path, "r") as file:
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

    pp.create_std_type(net, new_std_type, name="Test", element="line")
    # Step 2: Create connections (lines) based on the matrix
    for i, row in enumerate(connections):
        for j, connection in enumerate(row):
            if j > i and connection:  # Only create connections for one half of the matrix
                pp.create_line(net, from_bus=bus_ids[i], to_bus=bus_ids[j], length_km=0.02, std_type="Test")

    # Step 3: Add components to each bus
    for bus_name, comps in components.items():
        # Get bus id
        bus_idx = bus_ids[int(bus_name.replace("bus", ""))]

        # Create PV if exists in components
        if "pv" in comps:
            name = comps["pv"]
            pp.create_sgen(net, bus=bus_idx, p_mw=0.0, q_mvar=0.0, name=f"pv{bus_name}")

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
                max_e_mwh=storage_data["max_e_mwh"],
                # TODO: capacity of car battery is part of the car not the network. no need to define it here
                soc_percent=storage_data.get("soc", 100.0),  # Default value if not provided
                name=f"cs{bus_name}"
            )

    # Create the file name with the number of buses
    file_name = f'data/pandappower_networks/{base_name}_pp.json'

    # Save the network to JSON
    pp.to_json(net, filename=file_name)


def main():
    # Call your function here with the necessary arguments

    profile_name = 'simple_CS_noLoad_2bus'

    create_pp_network(profile_name)


if __name__ == "__main__":
    main()
