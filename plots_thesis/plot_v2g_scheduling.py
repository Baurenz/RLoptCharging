import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import matplotlib.lines as mlines
import plot_style  # Import the custom plot style module

# Define selected cars, model, session prefix, and ID
selected_cars = [1]
selected_model = 'DDPG'
session_prefix = "thesis_4busCs_nov2g_Load_pv_da"
session_id = 12
selected_episode = 180

plot_first_sub = True
plot_x_label = True

plot_style.set_plot_style()
plt.rcParams['font.size'] = 14  # Adjust based on your document's specific needs

base_path = f'../solvers/evaluation/Results/{selected_model}_{session_prefix}{session_id}'

# Define plot options for visibility control
plot_options = {
    'day_ahead_price': True,
    'soc_cs': True,
    'soc_ess': True,
    'renewable': True,
    'remaining_pv': False,
    'load': True,
    'power_ess': True,
    'power_cs': True,
    'power_ext_grid': True,
    'cost_ext_grid': False
}


# Function to load data from JSON file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Function to get available episodes
def get_available_episodes(file_path):
    episode_list = []
    for root, _, files in os.walk(file_path):
        for file in files:
            if file.startswith('results_') and file.endswith('.json'):
                episode_number = file[len('results_'):-5]
                episode_list.append(int(episode_number))
    return sorted(episode_list)


# Function to extract data from JSON
def extract_data_from_json(initial_values_json, results_json):
    soc_cs = np.array(results_json['soc_cs'])
    soc_cs[soc_cs == -1] = None

    soc_cs = np.round(soc_cs, 2) * 100
    num_cars = soc_cs.shape[0]

    power_cs = np.array(results_json['power_cs']).reshape(-1, num_cars).T
    present_cars = np.array(initial_values_json['present_cars'])

    present_cars = present_cars[:, :-1]  # Cut the last timestep for all cars
    # Update power_cs values to np.nan where present_cars is 0 for the corresponding car and timestep
    for car_idx in range(num_cars):
        power_cs[car_idx] = np.where(present_cars[car_idx] == 1, power_cs[car_idx], np.nan)

    departure_t = [np.array(dt) for dt in initial_values_json['departure_t']]
    arrival_t = [np.array(at) for at in initial_values_json['arrival_t']]

    soc_ess = np.array(results_json['soc_ess'])
    soc_ess = np.round(soc_ess, 2) * 100
    power_ess = np.array(results_json['power_ess']).reshape(-1, num_cars).T

    renewable_dict = results_json['renewable']
    renewable_power = np.array([renewable_dict[str(i)] for i in range(len(renewable_dict))])[:, :97]

    load_dict = results_json['load']
    load_power = np.array([load_dict[str(i)] for i in range(len(load_dict))])[:, :97] if load_dict else None

    day_ahead_price = np.array(results_json['day_ahead_episode'][:97])
    day_ahead_date = results_json['day_ahead_date']
    power_ext_grid = results_json['grid_final']
    cost_ext_grid = results_json['grid_cost']

    bus_energies = results_json['bus_energies']

    remaining_pv = np.zeros((renewable_power.shape[0], len(bus_energies)))

    # Loop through each timestep
    for timestep_idx, timestep in enumerate(bus_energies):
        # Make sure not to exceed the 97 timesteps limit
        if timestep_idx >= len(bus_energies):
            break

        # Loop through each bus within the current timestep
        for bus_idx in range(renewable_power.shape[0]):
            # Check if the bus data exists for the current timestep
            if str(bus_idx) in timestep:
                # Read 'res_available' for the current bus at the current timestep
                remaining_pv[bus_idx, timestep_idx] = timestep[str(bus_idx)]['pv_available']

    return renewable_power, remaining_pv, load_power, power_ess, power_cs, departure_t, arrival_t, day_ahead_price, day_ahead_date, soc_ess, soc_cs, cost_ext_grid, power_ext_grid


# Define label styles
label_styles = {
    'soc_cs': {'color': plot_style.color_cycle[0], 'linestyle': '--'},
    'power_cs': {'color': plot_style.color_cycle[0], 'linestyle': '-', 'drawstyle': 'steps-post'},
    'soc_ess': {'color': plot_style.color_cycle[1], 'linestyle': '--'},
    'power_ess': {'color': plot_style.color_cycle[1], 'linestyle': '-', 'drawstyle': 'steps-post'},
    'load': {'color': plot_style.color_cycle[2], 'linestyle': '-', 'drawstyle': 'steps-post'},
    'renewable': {'color': plot_style.color_cycle[3], 'linestyle': '-', 'drawstyle': 'steps-post'},
    'remaining_pv': {'color': plot_style.color_cycle[3], 'linestyle': ':', 'drawstyle': 'steps-post'},
    'day_ahead_price': {'color': 'black', 'linestyle': '-.', 'drawstyle': 'steps-post'},
    'power_ext_grid': {'color': plot_style.color_cycle[4], 'linestyle': '-', 'drawstyle': 'steps-post'},
    'cost_ext_grid': {'color': plot_style.color_cycle[5], 'linestyle': ':'}

}


# Plot data function
def plot_data(renewable_power, remaining_pv, load_power, power_ess, power_cs, departure_t, arrival_t, day_ahead_price, day_ahead_date,
              soc_ess, soc_cs, cost_ext_grid, power_ext_grid, plot_options):
    # Determine the number of subplots
    subplot_count = len(selected_cars) + (1 if plot_first_sub else 0)

    # height_per_subplot = 2.4  # Desired height per subplot in inches
    height_per_subplot = 3 # Desired height per subplot in inches

    total_height = height_per_subplot * subplot_count  # Total height of the figure
    width = 12  # Desired width of the figure in inches

    # Create the figure and subplots with the calculated dimensions
    fig, axs = plt.subplots(subplot_count, 1, figsize=(width, total_height))

    # if not plot_first_sub:
    #     fig, axs = plt.subplots(subplot_count, 1, figsize=(width, 2.5))
    # if plot_x_label:
    #     fig, axs = plt.subplots(subplot_count, 1, figsize=(width, 3))


    x_97 = np.linspace(0, 24, 97)
    major_ticks = np.arange(0, 25, 1)
    minor_ticks = np.linspace(0, 24, 97)

    def two_decimal_formatter(x, pos):
        return f'{x:.2f}'

    if plot_first_sub:
        # Day-ahead price plot
        if plot_options['day_ahead_price']:
            formatter = FuncFormatter(two_decimal_formatter)
            axs[0].plot(x_97, day_ahead_price, label='Day-Ahead Price', **label_styles['day_ahead_price'])
            # axs[0].set_title(f'Day-Ahead Price of {day_ahead_date}')
            axs[0].set_ylabel(r'Price in € / kWh')  # LaTeX-style formatting
            axs[0].set_xticks(major_ticks, labelbottom=False)
            axs[0].tick_params(axis='x', which='both', labelbottom=False)
            axs[0].set_xticks(minor_ticks, minor=True)
            axs[0].grid(which='both', linestyle='--', alpha=0.3, linewidth=0.5)
            axs[0].yaxis.set_major_formatter(formatter)
            axs[0].set_xlim([0, 24])  # Set x-axis limits to remove space before 0 and after 24

            handles, labels = axs[0].get_legend_handles_labels()

            if plot_options['power_ext_grid']:
                ax2_ext_grid = axs[0].twinx()
                ax2_ext_grid.plot(x_97, power_ext_grid, label='External Grid Power', **label_styles['power_ext_grid'])
                ax2_ext_grid.set_ylabel('Power (kW)')
                ax2_ext_grid.tick_params(axis='y')
                ax2_ext_grid.set_xticks(major_ticks, labelbottom=False)
                ax2_ext_grid.set_xticks(minor_ticks, minor=True)
                ax2_ext_grid.grid(which='both', linestyle=':', linewidth=0.5, alpha=0.3)

                handles2, labels2 = ax2_ext_grid.get_legend_handles_labels()
                handles.extend(handles2)
                labels.extend(labels2)
        # Plot external grid cost if enabled
        if plot_options['cost_ext_grid']:
            ax2 = axs[0].twinx()
            ax2.plot(x_97, cost_ext_grid, label='Cost Ext Grid', **label_styles['cost_ext_grid'])
            ax2.set_ylabel('Cost')

            handles3, labels3 = ax2.get_legend_handles_labels()
            handles.extend(handles3)
            labels.extend(labels3)

        ax2_ext_grid.legend(handles, labels, loc='upper right', fancybox=True, framealpha=0.9)

    car_subplot_start_idx = 1 if plot_first_sub else 0

    # Iterate through each car subplot
    for i, car_idx in enumerate(selected_cars):
        ax_idx = i + car_subplot_start_idx
        if subplot_count > 1:
            ax = axs[ax_idx]
        else:
            ax = axs  # For a single subplot, axs is not a list
        ax2 = ax.twinx()
        # Plot departure and arrival times

        for idx, t in enumerate(arrival_t[car_idx]):
            x_val = t / 4
            y_val = int(soc_cs[car_idx][t])
            ax.axvline(x=x_val, color='g', linestyle='--')
            ax2.plot(x_val, y_val, 'go')

            if idx == 0:  # Check if it is the first entry
                ax2.text(x_val + 0.15, y_val, f'{y_val}', color='green', verticalalignment='bottom', horizontalalignment='left')
            else:
                ax2.text(x_val - 0.15, y_val, f'{y_val}', color='green', verticalalignment='bottom', horizontalalignment='right')

        for idx, t in enumerate(departure_t[car_idx]):
            x_val = t / 4
            y_val = int(soc_cs[car_idx][t])
            ax.axvline(x=x_val, color='r', linestyle='--')
            ax2.plot(x_val, y_val, 'ro')

            if idx == len(departure_t[car_idx]) - 1:  # Check if it is the last entry
                ax2.text(x_val - 0.3, y_val - 2, f'{y_val}', color='red', verticalalignment='top', horizontalalignment='right')
            else:
                ax2.text(x_val + 0.15, y_val, f'{y_val}', color='red', verticalalignment='top', horizontalalignment='left')

            # Plot data series based on plot_options

        if plot_options['renewable']:
            ax.plot(x_97, renewable_power[car_idx], label='PV', **label_styles['renewable'])

        if plot_options['remaining_pv']:
            ax.plot(x_97, remaining_pv[car_idx], label='remaining PV', **label_styles['remaining_pv'])

        if plot_options['load'] and load_power is not None:
            ax.plot(x_97, load_power[car_idx], label='Load', **label_styles['load'])

        if plot_options['power_ess'] and power_ess.any():
            ax.plot(x_97, power_ess[car_idx][:97], label='Power ESS', **label_styles['power_ess'])

        if plot_options['power_cs']:
            ax.plot(x_97, power_cs[car_idx], label='Power CS', **label_styles['power_cs'])

        if plot_options['soc_cs']:
            ax2.plot(x_97, soc_cs[car_idx][:97], label=f'SOC CS Car {car_idx + 1}', **label_styles['soc_cs'])
            ax2.set_ylabel('SOC (%)')  # Set label and color for the SOC y-axis

            # Continue plotting other data on the primary axis
        if plot_options['soc_ess'] and soc_ess.any():
            ax2.plot(x_97, soc_ess[car_idx][:97], label=f'SOC ESS Car {car_idx + 1}', **label_styles['soc_ess'])

            # Set up axes and grids for the prel_stylesimary axis
        # ax.set_title(f'Car {car_idx + 1}')


        ax.set_ylabel('Power in kW')
        if plot_first_sub:
            # For the first subplot, disable x-axis labels
            ax.tick_params(axis='x', which='both', labelbottom=False)
        else:
            # For subsequent subplots, ensure x-axis labels are visible
            ax.tick_params(axis='x', which='both', labelbottom=True)

        if plot_x_label:
            ax.set_xlabel('Time of the Day (hours)')
            ax.tick_params(axis='x', which='both', labelbottom=True)

        else:
            ax.tick_params(axis='x', which='both', labelbottom=False)

        ax.set_xticks(minor_ticks, minor=True)
        ax.tick_params(axis='x', which='both', top=True)
        ax.set_xlim([0, 24])  # Set x-axis limits for car subplots to remove space before 0 and after 24
        ax.grid(which='both', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.set_ylim(-11.5, 14)
        ax.set_ylim(0, 14)


        # Disable the grid for the secondary y-axis (SOC axis)
        ax2.grid(False)  # This will disable the grid for the secondary y-axis

        # Collect labels and handles for the legend from both axes
        handles_ax, labels_ax = ax.get_legend_handles_labels()
        handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
        handles = handles_ax + handles_ax2
        labels = labels_ax + labels_ax2

        # Ensure the SOC axis has its own legend and grid
        ax.legend(handles, labels, loc='upper right', fancybox=True, framealpha=0.9)

    fig.tight_layout()
    # plt.savefig(f'./plots/scheduling_{selected_model}2.pdf', bbox_inches='tight', pad_inches=0)
    # plt.savefig(f'/home/laurenz/Documents/DAI/_Thesis/git/Thesis_template/Figures/plots/scheduling_{selected_model}_{session_prefix}.pdf')
    file_name = f'scheduling_{selected_model}_{session_prefix}_{session_id}_{selected_episode}'
    plot_style.save_fig(file_name)
    plt.show()


# Main function to control the flow
def main(file_path, plot_options):
    initial_episode = get_available_episodes(file_path)[selected_episode]
    initial_values_json = load_json(f'{file_path}/Initial_Values_{initial_episode}.json')
    results_json = load_json(f'{file_path}/results_{initial_episode}.json')
    data = extract_data_from_json(initial_values_json, results_json)
    plot_data(*data, plot_options)


# Example usage
file_path = base_path
main(file_path, plot_options)
