import numpy as np
import json

# Initialize a 11x11 matrix with zeros for a system with 11 buses (0 to 10)
matrix_size = 11
adjacency_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

# Define the connections based on the provided information
connections = {
    0: [0, 1],
    1: [0, 1, 2],
    2: [1, 2, 3, 7],
    3: [2, 3, 4],
    4: [3, 4],
    5: [5, 6],
    6: [5, 6, 7],
    7: [2, 6, 7, 8, 10],
    8: [7, 8, 9],
    9: [8, 9],
    10: [7, 10]
}

# Populate the adjacency matrix
for bus, connected_buses in connections.items():
    for connected_bus in connected_buses:
        adjacency_matrix[bus][connected_bus] = 1

# Convert the adjacency matrix to a list for JSON serialization
adjacency_list = adjacency_matrix.tolist()

# Prepare the data in the required format
data = {
    "buses": matrix_size,
    "connections": adjacency_list
}

# Save the data to a JSON file in the current directory
with open('bus_matrix.json', 'w') as json_file:
    json.dump(data, json_file)

print("JSON file saved successfully.")
