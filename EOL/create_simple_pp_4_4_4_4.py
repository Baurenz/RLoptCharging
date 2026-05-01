import pandapower as pp

# Create an empty network
net = pp.create_empty_network()

# Create buses
bus1 = pp.create_bus(net, vn_kv=0.4)
bus2 = pp.create_bus(net, vn_kv=0.4)
bus3 = pp.create_bus(net, vn_kv=0.4)
bus4 = pp.create_bus(net, vn_kv=0.4)

# Create an external grid connection
pp.create_ext_grid(net, bus=bus1, vm_pu=1.0)

# Create lines
pp.create_line(net, from_bus=bus1, to_bus=bus2, length_km=1.0, std_type="15-AL1/3-ST1A 0.4")
pp.create_line(net, from_bus=bus2, to_bus=bus3, length_km=1.0, std_type="15-AL1/3-ST1A 0.4")
pp.create_line(net, from_bus=bus3, to_bus=bus4, length_km=1.0, std_type="15-AL1/3-ST1A 0.4")
pp.create_line(net, from_bus=bus4, to_bus=bus1, length_km=1.0, std_type="15-AL1/3-ST1A 0.4")

# Create PVs
for idx, bus in enumerate([bus1, bus2, bus3, bus4], 1):
    pp.create_sgen(net, bus, p_mw=0.1, q_mvar=0.02, name=f"pv{idx}")

# Create batteries
for idx, bus in enumerate([bus1, bus2, bus3, bus4], 1):
    pp.create_storage(net, bus, p_mw=0.01, q_mvar=0, sn_mva=0, soc_percent=100.0, max_e_mwh=0.5,
                      name=f"battery{idx}")

# Create loads
for idx, bus in enumerate([bus1, bus2, bus3, bus4], 1):
    pp.create_load(net, bus, p_mw=0.1, q_mvar=0.02, name=f"load{idx}")

# Save the network to JSON
pp.to_json(net, filename='../data/scenarios/4-Bus_4-PV_4-Load_4-V2G_externalgrid_bus1_no_data.json')
