from datetime import datetime, date
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

# File paths
results_folder = 'rl_OptV2GEnv/files/Results'  # replace with your actual path
specific_results_file = ""  # add filename here if needed
# Get the latest run folder
run_folders = [f for f in os.listdir(results_folder) if f.startswith('Run-')]
run_folders.sort(reverse=True)  # sort in descending order
latest_run_folder = run_folders[0]

# Get the latest results JSON file from the run folder
results_files = [f for f in os.listdir(os.path.join(results_folder, latest_run_folder)) if
                 f.startswith('results') and f.endswith('.json')]
results_files.sort(reverse=True)  # sort in descending order
latest_results_file = results_files[0]


if specific_results_file:
    latest_results_file = specific_results_file

result_file_date = latest_results_file.split('-')[1:4] # split by '-' and get elements for year, month, and day
result_file_date = '-'.join(result_file_date) # join them back together with '-'
result_file_date = result_file_date.split('.')[0] # remove the file extension

# Read market data for the same date as the results file
market_data_path = f'data/market/{result_file_date}.csv'  # use date from results file
market_data = pd.read_csv(market_data_path)
electricity_price = market_data['marketprice'].values  # use 'marketprice' column

# Read results data
with open(os.path.join(results_folder, latest_run_folder, latest_results_file)) as json_file:
    results_data = json.load(json_file)

# Get the latest initial values JSON file from the run folder
initial_values_files = [f for f in os.listdir(os.path.join(results_folder, latest_run_folder)) if
                        f.startswith('Initial_Values-') and f.endswith('.json')]
initial_values_files.sort(reverse=True)  # sort in descending order
latest_initial_values_file = initial_values_files[0]

# If specific results file was provided, find matching initial values file
if specific_results_file:
    timestamp = specific_results_file.split('-')[1]  # extract timestamp from results filename
    matching_files = [f for f in initial_values_files if timestamp in f]
    if matching_files:
        latest_initial_values_file = matching_files[0]

# Read initial values data
with open(os.path.join(results_folder, latest_run_folder, latest_initial_values_file)) as json_file:
    initial_values_data = json.load(json_file)

# Extract arrival and departure times
arrival_times = initial_values_data['arrival_t']
departure_times = initial_values_data['departure_t']

# Get first and second entries from double arrays
charging_powers = results_data['charging_power']
first_entries = [powers[0] for powers in charging_powers]
second_entries = [powers[1] for powers in charging_powers]

# Other data
time_steps = range(24)  # assuming 24 hours
battery_socs = results_data['boc']
# since its 25 values, we leave the last one out. not sure if thats correct
battery_socs = [battery_soc[:-1] for battery_soc in battery_socs]

fig, axs = plt.subplots(3, 1, figsize=(10, 15))  # Create 3 vertical plots
cars = ['Car 1', 'Car 2']  # Car names for better readability

# Iterate over each car and plot the relevant data
for i, car in enumerate(cars):
    # Charging and discharging power for each car
    axs[i].plot(time_steps, [first_entries, second_entries][i], label=f'{car} Power')

    # Arrival and departure times
    for arrival_time in arrival_times[i]:
        axs[i].axvline(x=arrival_time, color='g', linestyle='--')  # green for arrival
    for departure_time in departure_times[i]:
        axs[i].axvline(x=departure_time, color='r', linestyle='--')  # red for departure

    axs[i].set_ylabel('Power (kW)')
    axs[i].legend()
    axs[i].grid(True)
    # TODO: optimal?!
    axs[i].set_title(f'Optimal Charging and Discharging Schedule for {car}')
    axs[i].set_xticks(time_steps)  # Add ticks at every hour

    # SoC of each car
    ax3 = axs[i].twinx()
    ax3.set_ylabel('SoC (%)', color='tab:red')
    ax3.plot(time_steps, [soc * 100 for soc in battery_socs[i]], color=['tab:blue', 'tab:orange'][i], linestyle='dashed', label=f'SoC {car}')
    ax3.tick_params(axis='y', labelcolor='tab:red')
    ax3.set_ylim(bottom=0)
    ax3.legend(loc='upper left')

# Market prices (displayed only once at the bottom)
axs[2].set_ylabel('Market Price (€/kWh)', color='tab:green')
axs[2].plot(time_steps, electricity_price, color='tab:green')
axs[2].tick_params(axis='y', labelcolor='tab:green')
axs[2].grid(True)
axs[2].set_title('Market Prices')
axs[2].set_xticks(time_steps)  # Add ticks at every hour

plt.suptitle(latest_run_folder, fontsize=16)

fig.tight_layout()
plt.show()
