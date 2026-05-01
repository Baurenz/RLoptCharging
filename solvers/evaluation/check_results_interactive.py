import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.widgets import CheckButtons, TextBox
import matplotlib.gridspec as gridspec

# Define session information
# Define selected cars
selected_cars = [0, 1]
selected_model = 'DDPG'
session_prefix = "epb150_pv200"
session_id = 1

global ax2_cost_ext_grid, ax2_power_ext_grid
ax2_cost_ext_grid = None
ax2_power_ext_grid = None


# Function to load data from JSON file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Function to get available episodes
def get_available_episodes(selected_model, session_prefix, session_id):
    episode_list = []
    base_path = f'Results/{selected_model}_{session_prefix}{session_id}/'

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.startswith('results_') and file.endswith('.json'):
                episode_number = file[len('results_'):-5]
                episode_list.append(int(episode_number))

    return sorted(episode_list)


# Function to extract data from JSON
def extract_data_from_json(initial_values_json, results_json):
    soc_cs = np.array(results_json['soc_cs'])
    soc_cs[soc_cs == -1] = None
    soc_ess = np.array(results_json['soc_ess'])

    num_cars = soc_cs.shape[0]
    renewable_dict = results_json['renewable']
    renewable_power = np.array([renewable_dict[str(i)] for i in range(len(renewable_dict))])[:, :97]

    load_dict = results_json['load']
    if load_dict:
        load_power = np.array([load_dict[str(i)] for i in range(len(load_dict))])[:, :97]
    else:
        load_power = None

    power_ess = np.array(results_json['power_ess']).reshape(-1, num_cars).T
    power_cs = np.array(results_json['power_cs']).reshape(-1, num_cars).T

    departure_t = [np.array(dt) for dt in initial_values_json['departure_t']]
    arrival_t = [np.array(at) for at in initial_values_json['arrival_t']]

    day_ahead_price = np.array(results_json['day_ahead_episode'][:97])
    day_ahead_date = results_json['day_ahead_date']

    power_ext_grid = results_json['grid_final']
    cost_ext_grid = results_json['grid_cost']

    return renewable_power, load_power, power_ess, power_cs, departure_t, arrival_t, day_ahead_price, day_ahead_date, soc_ess, soc_cs, cost_ext_grid, power_ext_grid


# Function to update the plot for a given episode
def update_episode(episode):
    try:
        episode = int(episode)  # Convert entered text to an integer
        if episode in available_episodes:
            # Load data for the new episode
            initial_values_json = load_json(f'Results/{selected_model}_{session_prefix}{session_id}/Initial_Values_{episode}.json')
            results_json = load_json(f'Results/{selected_model}_{session_prefix}{session_id}/results_{episode}.json')

            # Extract and plot new data
            data = extract_data_from_json(initial_values_json, results_json)
            plot_data(*data)

            # Update the legend and redraw the plot
            update_legend(label_styles)
            plt.draw()
        else:
            print(f"Episode {episode} is not available.")
    except ValueError:
        print("Invalid input. Please enter a valid episode number.")

    # Update the state of each checkbox
    for i, label in enumerate(checkbox_labels):
        current_state = check.get_status()[i]
        if current_state != visibility_states[label]:
            check.set_active(i)  # Toggle the state


# Function to toggle visibility of data on the plot
def update_plot(label):
    global ax2_cost_ext_grid, ax2_power_ext_grid

    graphs = {
        'soc_cs': lines_soc_cs,
        'soc_ess': lines_soc_ess,
        'renewable': lines_renewable,
        'load': lines_load,
        'power_ess': lines_power_ess,
        'power_cs': lines_power_cs,
        'power_ext_grid': line_power_ext_grid,
        'cost_ext_grid': line_cost_ext_grid,
        'day_ahead_price': line_day_ahead_price
    }

    visibility_states[label] = not visibility_states[label]

    for key, value in graphs.items():
        if label == key:
            for line in value:
                line.set_visible(visibility_states[key])

    if label == 'day_ahead_price':
        axs[0].get_yaxis().set_visible(visibility_states[label])
    elif label == 'cost_ext_grid' and ax2_cost_ext_grid is not None:
        ax2_cost_ext_grid.set_visible(visibility_states[label])
    elif label == 'power_ext_grid' and ax2_power_ext_grid is not None:
        ax2_power_ext_grid.set_visible(visibility_states[label])

    update_legend(label_styles)
    plt.draw()


# Function to update the legend based on visible lines
def update_legend(label_styles):
    labels = set()
    all_lines = lines_soc_cs + lines_soc_ess + lines_renewable + lines_load + lines_power_ess + lines_power_cs

    for line in all_lines:
        if line.get_visible():
            label = line.get_label().split(' ')[0]  # Adjust this to correctly parse your labels
            labels.add(label)

    legend_lines = [mlines.Line2D([], [], color=label_styles[label]['color'], linestyle=label_styles[label]['linestyle'])
                    for label in labels]
    legend_ax.legend(legend_lines, labels, loc='upper center', ncol=3)  # Adjust 'ncol' as needed
    legend_ax.axis('off')  # Hide the axis


# Get available episodes
available_episodes = get_available_episodes(selected_model, session_prefix, session_id)

# Define label styles
label_styles = {
    'soc_cs': {'color': 'red', 'linestyle': '--'},
    'soc_ess': {'color': 'cyan', 'linestyle': '--'},
    'renewable': {'color': 'purple', 'linestyle': '-'},
    'load': {'color': 'black', 'linestyle': '-'},
    'power_ess': {'color': 'cyan', 'linestyle': '-'},
    'power_cs': {'color': 'red', 'linestyle': '-'}
}

visibility_states = {
    'soc_cs': True,
    'soc_ess': True,
    'renewable': True,
    'load': True,
    'power_ess': True,
    'power_cs': True,
    'power_ext_grid': True,
    'cost_ext_grid': True,
    'day_ahead_price': True
}

# Initialize lists for plot lines
lines_soc_cs = []
lines_soc_ess = []
lines_renewable = []
lines_load = []
lines_power_ess = []
lines_power_cs = []
line_power_ext_grid = []
line_cost_ext_grid = []
line_day_ahead_price = []
# Define x-axis values

x_97 = np.linspace(0, 24, 97)

fig = plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
gs = gridspec.GridSpec(len(selected_cars) + 2, 2, width_ratios=[1, 10], hspace=0.5, wspace=0.1)  # Add an extra row for legend

# Define axes for the plots as before
axs = [fig.add_subplot(gs[i, 1]) for i in range(len(selected_cars) + 1)]

# Create a new axis for the legend, adjust the row index as per your layout
legend_ax = fig.add_subplot(gs[len(selected_cars) + 1, :])  # Spanning the entire row

plt.subplots_adjust(top=0.97, bottom=0.03, left=0.03, right=0.97, hspace=0.05, wspace=0.05)

twin_axes = []


# Create a function to plot the data
def plot_data(renewable_power, load_power, power_ess, power_cs, departure_t, arrival_t, day_ahead_price, day_ahead_date, soc_ess, soc_cs,
              cost_ext_grid, power_ext_grid):
    global lines_soc_cs, lines_soc_ess, lines_renewable, lines_load, lines_power_ess, lines_power_cs, line_power_ext_grid, line_cost_ext_grid, line_day_ahead_price
    global ax2_cost_ext_grid, ax2_power_ext_grid

    # Initialize lists for plot lines
    lines_soc_cs = []
    lines_soc_ess = []
    lines_renewable = []
    lines_renewable = []
    lines_load = []
    lines_power_ess = []
    lines_power_cs = []
    line_power_ext_grid = []

    global twin_axes
    # Clear existing twin axes
    for tax in twin_axes:
        tax.remove()  # Change from clear() to remove()
    twin_axes = []
    axs[0].clear()
    line_day_ahead_price = [axs[0].plot(x_97, day_ahead_price, label='Day-Ahead Price', color='purple')[0]]
    axs[0].set_title(f'Day-Ahead Price of {day_ahead_date}')
    axs[0].set_ylabel('Price (Currency)', color='purple')
    axs[0].tick_params(axis='y', labelcolor='purple')
    axs[0].set_xlim([0, 24])

    # Set x-axis ticks to be hourly
    hourly_ticks = np.linspace(0, 24, 25)  # Generates 25 points from 0 to 24 (inclusive), one for each hour
    axs[0].set_xticks(hourly_ticks)  # Set the ticks to be at these hourly points

    minor_ticks = x_97  # Assuming 97 points for 24 hours
    axs[0].set_xticks(minor_ticks, minor=True)

    # Enable grid for major and minor ticks; minor ticks get the subtle grid
    axs[0].grid(True, which='major', linestyle='-', linewidth=0.5, color='gray', alpha=0.7)  # More prominent grid for hourly ticks
    axs[0].grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)  # Subtle grid for 15-minute intervals

    if line_cost_ext_grid is not None:
        for line in line_cost_ext_grid:
            line.remove()
    line_cost_ext_grid = []
    if ax2_cost_ext_grid is not None:
        ax2_cost_ext_grid.remove()

    # Create an additional axis for cost_ext_grid on the left
    ax2_cost_ext_grid = axs[0].twinx()
    # Shift the additional y-axis to the left
    ax2_cost_ext_grid.spines["right"].set_position(("axes", -0.3))
    ax2_cost_ext_grid.spines["right"].set_visible(True)
    ax2_cost_ext_grid.spines["left"].set_visible(False)
    ax2_cost_ext_grid.yaxis.set_label_position('left')
    ax2_cost_ext_grid.yaxis.set_ticks_position('left')

    line_cost_ext_grid = [ax2_cost_ext_grid.plot(x_97, cost_ext_grid, label='Cost Ext Grid', color='green', linestyle=':')[0]]
    ax2_cost_ext_grid.set_ylabel('Cost (€)', color='green')
    ax2_cost_ext_grid.tick_params(axis='y', labelcolor='green')

    # Create twin axis for power_ext_grid
    ax2_power_ext_grid = axs[0].twinx()
    line_power_ext_grid = [
        ax2_power_ext_grid.plot(x_97, power_ext_grid, label='Power Ext Grid', color='blue', linestyle='-.')[0]]  # as a list
    ax2_power_ext_grid.set_ylabel('Power (kW)')
    ax2_power_ext_grid.tick_params(axis='y', labelcolor='blue')
    twin_axes.append(ax2_power_ext_grid)

    for i, car_idx in enumerate(selected_cars):
        ax = axs[i + 1]

        ax.clear()

        for t in departure_t[car_idx]:
            x_val = t / 4
            y_val = soc_cs[car_idx][t]
            ax.axvline(x=x_val, color='r', linestyle='--')
            ax.plot(x_val, y_val, 'ro')
            ax.text(x_val, y_val, f'{y_val:.2f}', color='red', verticalalignment='bottom', horizontalalignment='right')

        for t in arrival_t[car_idx]:
            x_val = t / 4
            y_val = soc_cs[car_idx][t]
            ax.axvline(x=x_val, color='g', linestyle='--')
            ax.plot(x_val, y_val, 'go')
            ax.text(x_val, y_val, f'{y_val:.2f}', color='green', verticalalignment='bottom', horizontalalignment='left')

        ax2 = ax.twinx()
        twin_axes.append(ax2)
        ax2.set_ylabel('Energy (kW)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        ax.set_title(f'Car {car_idx + 1}')
        ax.set_xlabel('Time (15-minute intervals)')
        ax.set_ylabel('soc')
        ax.set_xlim([0, 24])

        # Set major ticks to hourly intervals
        ax.set_xticks(hourly_ticks)

        # Add minor ticks for every 15-minute interval
        minor_ticks = x_97  # Assuming 97 points for 24 hours
        ax.set_xticks(minor_ticks, minor=True)

        # Enable grid for major and minor ticks; minor ticks get the subtle grid
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray', alpha=0.7)  # More prominent grid for hourly ticks
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)  # Subtle grid for 15-minute intervals

        lines_soc_cs.append(axs[i + 1].plot(x_97, soc_cs[car_idx][:97], label=f'soc_cs Car {car_idx + 1}', **label_styles['soc_cs'])[0])
        if soc_ess != 0:
            lines_soc_ess.append(
                axs[i + 1].plot(x_97, soc_ess[car_idx][:97], label=f'soc_ess Car {car_idx + 1}', **label_styles['soc_ess'])[0])
            lines_power_ess.append(ax2.plot(x_97, power_ess[car_idx][:97], label=f'power_ess', **label_styles['power_ess'])[0])

        lines_renewable.append(ax2.plot(x_97, renewable_power[car_idx], label=f'renewable', **label_styles['renewable'])[0])
        if load_power:
            lines_load.append(ax2.plot(x_97, load_power[car_idx], label=f'load', **label_styles['load'])[0])

        lines_power_cs.append(ax2.plot(x_97, power_cs[car_idx], label=f'power_cs', **label_styles['power_cs'])[0])

        for line_list, state in zip([lines_soc_cs, lines_soc_ess, lines_renewable, lines_load, lines_power_ess, lines_power_cs],
                                    visibility_states.values()):
            for line in line_list:
                line.set_visible(state)

    return ax, axs, ax2_cost_ext_grid, ax2_power_ext_grid


# Plot the initial data
initial_episode = available_episodes[0]
initial_values_json = load_json(f'Results/{selected_model}_{session_prefix}{session_id}/Initial_Values_{initial_episode}.json')
results_json = load_json(f'Results/{selected_model}_{session_prefix}{session_id}/results_{initial_episode}.json')

data = extract_data_from_json(initial_values_json, results_json)

ax, axs, ax2_cost_ext_grid, ax2_power_ext_grid = plot_data(*data)

update_legend(label_styles)

# Create checkboxes to toggle data visibility
checkbox_labels = ['day_ahead_price', 'soc_cs', 'soc_ess', 'renewable', 'load', 'power_ess', 'power_cs', 'power_ext_grid', 'cost_ext_grid']

# Define axes for the widgets
checkbox_ax = fig.add_subplot(gs[0:2, 0])  # Adjust grid range as needed
text_box_ax = fig.add_subplot(gs[2, 0])  # Adjust grid range as needed

# Create checkboxes and text box
check = CheckButtons(checkbox_ax, checkbox_labels, [True] * len(checkbox_labels))
text_box = TextBox(text_box_ax, 'Ep: ')

# Connect the callbacks
check.on_clicked(update_plot)
text_box.on_submit(update_episode)

# Create a text box for entering episode number

plt.tight_layout(pad=3, w_pad=0.5, h_pad=1.0)
plt.show()
