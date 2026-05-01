import json
import random

def create_json(buses=30):
    # Build connections matrix
    connections = [[0]*buses for _ in range(buses)]
    for i in range(buses):
        connections[i][i] = 1
        if i+1 < buses:
            connections[i][i+1] = 1
        if i-1 >= 0:
            connections[i][i-1] = 1
    connections[0][-1] = 1
    connections[-1][0] = 1

    # Build components dict
    components = {}
    pvs = ['pv1', 'pv2']
    loads = ['load1', 'load2']
    for i in range(buses):
        bus_name = f'bus{i}'
        components[bus_name] = {
            'pv': random.choice(pvs),
            'load': random.choice(loads)
        }

    # Construct the full JSON structure
    data = {
        "buses": buses,
        "connections": connections,
        "components": components,
        "ext_grid_bus": "bus0"
    }

    return data

if __name__ == "__main__":
    buses_num = int(input("Enter number of buses: "))
    data = create_json(buses_num)
    filename = f'dynamic_profile_{buses_num}bus.json'
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=4)
    print(f'JSON saved to "{filename}".')