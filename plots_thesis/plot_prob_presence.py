import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.dates as mdates


import plot_style

# Parameters
n_cs = 10  # Number of charging stations
last_timestep = 97  # Last timestep (for a full day divided into 15-minute intervals)
plot_style.set_plot_style()


def reset_init_cs_presence_soc(last_timestep, n_cs):
    arrival_probabilities = [(0, 0.85), (8, 0.1), (32, 0.05), (48, 0.1), (64, 0.15), (80, 0.15), (96, 0.15)]
    departure_probabilities = [(0, 0.04), (28, 0.2), (36, 0.1), (48, 0.15), (64, 0.1), (80, 0.1), (96, 0.1)]

    soc_cs_init = - np.ones([n_cs, last_timestep + 1])
    present_cars = np.zeros([n_cs, last_timestep + 1])
    arrival_t = []
    departure_t = []

    def get_probability(probabilities, hour):
        for time, prob in reversed(probabilities):
            if hour >= time:
                return prob
        return 0

    for car in range(n_cs):
        present = False
        arrival_car = []
        departure_car = []
        arrival_hour = None  # Variable to track the arrival hour

        for hour in range(last_timestep):
            if not present and hour < last_timestep - 2:  # Preventing arrivals in the last two timesteps
                arrival_prob = get_probability(arrival_probabilities, hour)
                present = random.random() < arrival_prob
                if present:
                    soc_cs_init[car, hour] = random.randint(20, 60) / 100
                    arrival_car.append(hour)
                    arrival_hour = hour  # Set the arrival hour when the car arrives

            elif present:
                # Check for departure only if at least 6 timesteps have passed since arrival
                if hour >= arrival_hour + 14:
                    departure_prob = get_probability(departure_probabilities, hour)
                    will_leave = random.random() < departure_prob
                    if will_leave:
                        departure_car.append(hour)
                        present = False
                        arrival_hour = None  # Reset the arrival hour

            present_cars[car, hour] = 1 if present else 0

        # Ensure departure at the last timestep if still present
        if present:
            present_cars[car, last_timestep] = 0
            departure_car.append(last_timestep)

        arrival_t.append(arrival_car)
        departure_t.append(departure_car)

    evolution_of_cars = np.sum(present_cars, axis=0)

    return soc_cs_init, arrival_t, departure_t, evolution_of_cars, present_cars


# Initialize an array to store the sum of present cars at each timestep over all simulations
total_present_cars = np.zeros(last_timestep + 1)

# Run the simulation 1000 times
for _ in range(1000):
    _, _, _, _, present_cars = reset_init_cs_presence_soc(last_timestep, n_cs)
    total_present_cars += np.sum(present_cars, axis=0)  # Sum up the presence of cars at each timestep

# Calculate the probability of a car being present at each timestep
probability_presence = total_present_cars / (n_cs * 1000)

# Convert timesteps to hours for plotting
time_hours = np.arange(last_timestep + 1) / 4  # Assuming 4 timesteps per hour (15-minute intervals)

# Create a figure and axis for plotting
fig, ax = plt.subplots()

# Plotting with filled area under the curve
ax.fill_between(time_hours, probability_presence, label='Probability of Car Presence', step='pre', alpha=0.9, color=plot_style.color_cycle[2])

# Set x-axis major ticks to be every 3 hours
ax.xaxis.set_major_locator(plt.MultipleLocator(3))
# Minor ticks every hour
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

# Generate labels for every third hour
time_labels = [f'{int(hour):02d}:00' for hour in np.arange(0, 25, 3)]  # Every third hour

# Set x-axis major tick labels
ax.set_xticks(np.arange(0, 25, 3))  # Ensure ticks are placed at every third hour
ax.set_xticklabels(time_labels)

# Set x-axis and y-axis limits to cut the plot at "00:00" and "24:00"
ax.set_xlim(0, 24)
ax.set_ylim(bottom=0.2)  # Ensure the y-axis starts at 0

# Adding grid lines
ax.grid(True, which='major', linestyle='--')  # Major grid lines
ax.grid(True, which='minor', linestyle='--')  # Minor grid lines without labels

# Adding labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Probability')
# ax.set_title('Probability of Car Presence at Charging Stations Over Time')

# Add legend
ax.legend()

plt.savefig(f'./plots/presence_car_prob.pdf')
plt.savefig(f'/home/laurenz/Documents/DAI/_Thesis/git/Thesis_template/Figures/plots/presence_car_prob.pdf')

# Show plot
plt.show()
