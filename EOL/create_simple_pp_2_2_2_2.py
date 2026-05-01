import pandapower as pp

# Create an empty network
net = pp.create_empty_network()

# Create buses
bus1 = pp.create_bus(net, vn_kv=0.4)
bus2 = pp.create_bus(net, vn_kv=0.4)

# Create an external grid connection
pp.create_ext_grid(net, bus=bus1, vm_pu=1.0)

# Create a line between the two buses
pp.create_line(net, from_bus=bus1, to_bus=bus2, length_km=1.0, std_type="15-AL1/3-ST1A 0.4")

# Create PV and battery for each bus and connect them to their respective buses
pv1 = pp.create_sgen(net, bus1, p_mw=0.1, q_mvar=0.02, name="PV1")
pv2 = pp.create_sgen(net, bus2, p_mw=0.1, q_mvar=0.02, name="PV2")

# Assume a battery capacity of 0.5 MWh
battery1 = pp.create_storage(net, bus1, p_mw=0.01, q_mvar=0, sn_mva=0, soc_percent=100.0, max_e_mwh=0.5,
                             name="Battery1")
battery2 = pp.create_storage(net, bus2, p_mw=-0.05, q_mvar=0, sn_mva=0, soc_percent=100.0, max_e_mwh=0.5,
                             name="Battery2")

# Create consumers
load1 = pp.create_load(net, bus1, p_mw=0.1, q_mvar=0.02, name="Load1")
load2 = pp.create_load(net, bus2, p_mw=0.1, q_mvar=0.02, name="Load2")

# Save the network to JSON
pp.to_json(net, filename='../data/scenarios/2-Bus_2-PV_2-Load_2-V2G_externalgrid_bus1_no_data.json')
