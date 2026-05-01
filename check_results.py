import matplotlib.pyplot as plt
import numpy as np
import json

# Function to load data from JSON file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Load the .json files
initial_values_json = load_json('rl_OptV2GEnv/files/Results/Run-1706084889/Initial_Values-2024-01-24-09-33-45.json')
results_json = load_json('rl_OptV2GEnv/files/Results/Run-1706084889/results-2024-01-24-09-33-45.json')

# Access the 'BOC' and 'DepartureT' fields from the JSON data
soc_cs = np.array(results_json['soc_cs'])  # Convert to numpy array if needed
soc_cs[soc_cs == 0] = None

soc_ess = np.array(results_json['soc_ess'])

# Extract renewable/ load data and convert it into a numpy array
renewable_dict = results_json['renewable']
renewable = np.array([renewable_dict[str(i)] for i in range(len(renewable_dict))])
renewable = renewable[:, :97]  # Truncate renewable data to 97 timesteps

load_dict = results_json['load']
load = np.array([load_dict[str(i)] for i in range(len(load_dict))])
load = load[:, :97]  # Truncate load data to 97 timesteps

# Convert departure and arrival times to list of numpy arrays
departure_t = [np.array(dt) for dt in initial_values_json['departure_t']]
arrival_t = [np.array(at) for at in initial_values_json['arrival_t']]

# Extract day-ahead price data
day_ahead_price = np.array(results_json['day_ahead_episode'][:97])

# Define the x-axis for 97 points
x_97 = np.linspace(0, 24, 97)

# Get the number of cars from the 'boc' data
num_cars = soc_cs.shape[0]

# Create a figure with a number of subplots matching the number of cars + 1 for day-ahead price
fig, axs = plt.subplots(num_cars + 1, 1, figsize=(10, 2 * (num_cars + 1)))

# Plot the day-ahead price on the first subplot
axs[0].plot(x_97, day_ahead_price, label='Day-Ahead Price', color='purple')
axs[0].set_title('Day-Ahead Price')
axs[0].set_ylabel('Price')
axs[0].set_xlim([0, 24])
axs[0].legend()

# For each car, plot the data
for i in range(num_cars):
    # Use the correct subplot for each car
    ax = axs[i + 1]

    # Plot the 'BOC' data
    ax.plot(x_97, soc_cs[i], label='soc_cs')
    ax.plot(x_97, soc_ess[i], label='soc_ess')

    # Plot vertical lines and markers with text at the departure times
    for t in departure_t[i]:
        x_val = t / 4  # Convert to hours
        y_val = soc_cs[i][t]  # Get the corresponding BOC value
        ax.axvline(x=x_val, color='r', linestyle='--')
        ax.plot(x_val, y_val, 'ro')  # Red marker for departure
        ax.text(x_val, y_val, f'{y_val:.2f}', color='red', verticalalignment='bottom', horizontalalignment='right')

    # Plot vertical lines and markers with text at the arrival times
    for t in arrival_t[i]:
        x_val = t / 4  # Convert to hours
        y_val = soc_cs[i][t]  # Get the corresponding BOC value
        ax.axvline(x=x_val, color='g', linestyle='--')
        ax.plot(x_val, y_val, 'go')  # Green marker for arrival
        ax.text(x_val, y_val, f'{y_val:.2f}', color='green', verticalalignment='bottom', horizontalalignment='left')

    # Create a secondary y-axis for renewable production
    ax2 = ax.twinx()
    ax2.plot(x_97, renewable[i], label='Renewable Production', color='blue', linestyle='--')
    ax2.plot(x_97, load[i], label='Load', color='orange', linestyle='-.')  # Plot load data
    ax2.set_ylabel('Energy (kW)', color='blue')  # Adjust the ylabel to 'Energy' for both renewable and load
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.legend(loc='upper right')

    # Set the title and labels for the primary axis
    ax.set_title(f'Car {i + 1}')
    ax.set_xlabel('Time (15-minute intervals)')
    ax.set_ylabel('soc')
    ax.set_xlim([0, 24])
    ax.legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
