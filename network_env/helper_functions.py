from datetime import datetime
import json

import pandapower as pp
import pandas as pd

from pandapower.timeseries import DFData, OutputWriter


def load_network_config(network_config_path):
    # Load the data from the JSON file
    with open(network_config_path, 'r') as f:
        network_config = json.load(f)

    # Extract the number of buses
    n_bus = network_config["buses"]

    n_ess = sum('ess' in bus for bus in network_config["components"].values())
    n_cs = sum('cs' in bus for bus in network_config["components"].values())
    n_pv = sum('pv' in bus for bus in network_config["components"].values())
    n_load = sum('load' in bus for bus in network_config["components"].values())

    # Initialize lists for locations of ess and cs

    loc_ess = [int('ess' in network_config["components"][f"bus{i}"]) for i in range(n_bus)]
    loc_cs = [int('cs' in network_config["components"][f"bus{i}"]) for i in range(n_bus)]

    return network_config, n_bus, n_pv, n_load, n_ess, loc_ess, n_cs, loc_cs


def load_pp_network(pp_network_path, bus_dict):
    net = pp.from_json(pp_network_path)
    # data_path = "../../data/dynamic_profiles/20000/"

    type_mapping = {
        'sgen': 'pv',
        'load': 'load'
    }

    # Load profiles based on component names in the network
    # for element_type, prefix in type_mapping.items():
    #     elements_df = getattr(net, element_type)  # like net.sgen, net.load
    #
    #     # Determine how the elements are associated with buses.
    #     for idx, row in elements_df.iterrows():
    #         # Using the bus value to construct the filename
    #         data_name = f"bus{row['bus']}_{prefix}"
    #         df = pd.read_csv(f"{data_path}{data_name}.csv", index_col=0)
    #         df.index = pd.to_datetime(df.index).time

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'./results/{timestamp}'
    ow = OutputWriter(net, output_path=output_path, output_file_type=".csv")

    # idx_dict = {}
    # for element_type, prefix in type_mapping.items():
    #     elements_df = getattr(net, element_type)
    #     for idx, row in elements_df.iterrows():
    #         key = f"{prefix}bus{row['bus']}"  # Adjusted key to match the new naming convention
    #         idx_dict[key] = idx

    # Define what to output
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_sgen', 'p_mw')
    ow.log_variable('res_storage', 'p_mw')
    ow.log_variable('res_ext_grid', 'p_mw')
    ow.log_variable('res_line', 'p_from_mw')

    return net#, idx_dict


def print_initial_values(net):
    # Extract relevant data
    sgen_data = net.sgen[['name', 'bus', 'p_mw']].set_index('bus')
    load_data = net.load[['name', 'bus', 'p_mw']].set_index('bus')
    storage_data = net.storage[['name', 'bus', 'p_mw']].set_index('bus')

    # Rename columns for clarity
    sgen_data = sgen_data.rename(columns={'p_mw': 'sgen_p_mw'})
    load_data = load_data.rename(columns={'p_mw': 'load_p_mw'})
    storage_data = storage_data.rename(columns={'p_mw': 'storage_p_mw'})

    # Merge the dataframes on the 'bus' column
    merged_data = pd.concat([sgen_data, load_data, storage_data], axis=1)

    # Print the consolidated data
    print(merged_data)


def run_single_timestep_pf(instance, time_step, idx_dict, bus_dict, loc_cs, loc_ess, cs_action_p, ess_action_p):
    # TODO: clean function paramters, way too many not needed
    # For every element in idx_dict, fetch the data and set the net values

    # TODO: get rid of this probably legacy.
    net = instance.net

    action_index_cs = 0
    action_index_ess = 0

    for i, bus in bus_dict.items():

        # TODO Next: Why v2g_action array of 5?? only 4 charging stations
        if bus.has_pv:
            # Construct the sgen name using the bus index
            sgen_name = f'pvbus{i}'
            # Find the index of the sgen element with the matching name
            sgen_index = net.sgen[net.sgen.name == sgen_name].index.tolist()

            # Check if the sgen element exists
            if sgen_index:
                # Retrieve the production value for the current timestep
                pv_value = bus.pv.pv_prod_episode[time_step] / 1000

                net.sgen.at[sgen_index[0], 'p_mw'] = pv_value  # TODO: check if kW --> MW conversion is correct
            else:
                print("EXIT: MISSMATCH in Configuration and to be simulated Network")
                exit()

        if bus.has_load:

            load_name = f'loadbus{i}'

            load_index = net.load[net.load.name == load_name].index.tolist()

            if load_index:
                load_value = bus.load.load_episode[time_step] / 1000  # conversion to kW

                net.load.at[load_index[0], 'p_mw'] = load_value  # TODO Next: check if correct.
            else:
                print("EXIT: MISSMATCH in Configuration and to be simulated Network")
                exit()

        if bus.has_ess:
            # TODO: check if ess works correctly. too many reexamples without ess

            ess_name = f'essbus{i}'

            ess_index = net.storage[net.storage.name == ess_name].index.tolist()

            if ess_index:
                ess_index = ess_index[0]
                ess_value = ess_action_p[action_index_ess] / 1000  # conversion to kW

                # Update the pandapower storage element with the v2g action value
                # Assuming the index in v2g_actions corresponds to the bus index

                net.storage.at[ess_index, 'p_mw'] = ess_value  # converting the action value if necessary

                action_index_ess += 1
            else:
                print("EXIT: MISSMATCH in Configuration and to be simulated Network")
                exit()

        if bus.has_cs:
            # Construct the storage name using the bus index
            storage_name = f'csbus{i}'

            # Find the index of the storage element with the matching name
            cs_index = net.storage[net.storage.name == storage_name].index.tolist()

            # Check if the storage element exists
            if cs_index:
                # Get the first (and should be the only) index in the list
                cs_index = cs_index[0]
                cs_value = cs_action_p[action_index_cs] / 1000  # conversion to kW
                # Update the pandapower storage element with the v2g action value
                # Assuming the index in v2g_actions corresponds to the bus index
                net.storage.at[cs_index, 'p_mw'] = cs_value

                action_index_cs += 1

            else:
                print("EXIT: MISSMATCH in Configuration and to be simulated Network")
                exit()

        if instance.debug_flag and i == 0 and time_step == 48:
            pv_value_bus_48 = pv_value
            load_value_bus_48 = load_value
            cs_value_bus_48 = cs_value
            ess_value_bus_48 = ess_value

            # print(f"net_energy_bus: {net_energy_bus}")
            print("---Input Pandapower-----")
            print(f"for Bus: {i} @ timestep {time_step}")
            print(f"pv_prod_bus: {pv_value}")
            print(f"load_bus: {load_value}")
            print(f"p_cs: {cs_value}")
            print(f"p_ess: {ess_value}")
            print(f"net energy calculated: {load_value + cs_value + ess_value - pv_value}")
            print("----------------------------")

    try:
        pp.runpp(net, algorithm='nr', init='results')
    except Exception as e:
        print(f"Caught an exception: {e}")
    #
    if instance.debug_flag and time_step == 48:
        bus_id = 0

        print_bus_results(net, bus_id, time_step, pv_value_bus_48, load_value_bus_48, cs_value_bus_48, ess_value_bus_48)

    # net.res_bus considers ext_grid within power balance calculation. as I want to look at the buses individually, I want to cancel
    # that out. therefore distinct calculation for bus 0 and others.
    net.res_bus['p_mw'][0] = net.res_bus['p_mw'][0] + net.res_ext_grid['p_mw'][0]

    line_load = net.res_line['loading_percent'].values
    # TODO: might be easier to simply return net.res
    net_result_dict = {
        'time_step': time_step,
        'res_bus': net.res_bus,
        'res_load': net.res_load,
        'res_sgen': net.res_sgen,
        'res_storage': net.res_storage,
        'res_ext_grid': net.res_ext_grid,
        'res_line': line_load
    }

    return net_result_dict, net


def sum_power_results_for_bus(net, bus_id, pv_value_bus0, load_value_bus0, cs_value_bus0, ess_value_bus0):
    # Sum powers from static generators (sgen) connected to the bus
    sgen_sum = net.res_sgen.loc[net.sgen['bus'] == bus_id, 'p_mw'].sum()

    # Sum powers from loads connected to the bus
    load_sum = net.res_load.loc[net.load['bus'] == bus_id, 'p_mw'].sum()

    storage_sum_input = ess_value_bus0 + cs_value_bus0

    # Sum powers from storage units connected to the bus
    storage_sum = net.res_storage.loc[net.storage['bus'] == bus_id, 'p_mw'].sum()

    # Total sum of powers

    if bus_id < len(net.res_line):
        line_outgoing = net.res_line['p_from_mw'][bus_id]

    else:
        line_outgoing = 0

    if bus_id != 0:
        line_incoming = net.res_line['p_to_mw'][bus_id - 1]
        print(f"net.res_line to bus: {line_incoming}")

    ext_grid = net.res_ext_grid['p_mw'][0]

    print(f"net.res_line from bus: {line_outgoing}")

    print(f"net.res_ext_grid: {ext_grid}")

    # ATTENTION: distinction between bus0 and all others only makes sense if external grid connected to bus0.
    if bus_id == 0:
        # net.res_bus considers ext_grid within power balance calculation. as I want to look at the buses individually, I want to cancel
        # that out. therefore distinct calculation for bus 0 and others.
        manual_sum = load_sum + storage_sum - sgen_sum  # - ext_grid
        manual_sum_connected = load_sum + storage_sum + line_outgoing - sgen_sum - ext_grid
        res_bus = net.res_bus['p_mw'][bus_id] + ext_grid

    else:
        manual_sum = load_sum + storage_sum - sgen_sum
        manual_sum_connected = load_sum + storage_sum - sgen_sum - line_outgoing - line_incoming
        res_bus = net.res_bus['p_mw'][bus_id]

    return manual_sum, manual_sum_connected, res_bus


def print_bus_results(net, bus_id, time_step, pv_value_bus0, load_value_bus0, cs_value_bus0, ess_value_bus0):
    load_consumptions = net.res_load[net.load.bus == bus_id]
    print("\nLoad Consumptions:")
    print(load_consumptions[['p_mw', 'q_mvar']])

    sgen_contributions = net.res_sgen[net.sgen.bus == bus_id]
    print("\nStatic Generator Contributions:")
    print(sgen_contributions[['p_mw', 'q_mvar']])

    manual_sum, manual_sum_connected, res_bus = sum_power_results_for_bus(net=net, bus_id=bus_id, pv_value_bus0=pv_value_bus0,
                                                                          load_value_bus0=load_value_bus0,
                                                                          cs_value_bus0=cs_value_bus0,
                                                                          ess_value_bus0=ess_value_bus0)

    print("\n---Output Pandapower------")
    print(f"for Bus: {bus_id} @ timestep {time_step}")
    print(f"net.res_bus pandapower: {res_bus}")
    print(f"net.res sum: {manual_sum}")
    print(f"net.res sum w/ line: {manual_sum_connected}")
    print("----------------------------")
