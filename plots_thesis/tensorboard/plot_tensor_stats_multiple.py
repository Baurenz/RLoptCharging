import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

from plots_thesis import plot_style

plot_style.set_plot_style()
plt.rcParams['font.size'] = 15  # Adjust based on your document's specific needs

# Path to your logs directory

model_type = 'DDPG'
session_prefix = 'reward_shiftnorm_price_2bus_epb15_ss600_bus_costs'
session_id = 1

log_base = '../../solvers/results/logs/'
model_dir = f'{model_type}/{session_prefix}{session_id}/{model_type}_0'
log_dir = log_base + model_dir

# List all .tfevents files in the log directory and subdirectories
log_files = glob.glob(os.path.join(log_dir, '**', '*.tfevents.*'), recursive=True)

# Sort files by creation time
log_files.sort(key=os.path.getmtime)

# Define the tags you want to plot
tags_to_plot = {
    'custom/buying_ext_grid_mean_2048': 'Penalty Buying from Grid',
    'custom/penalty_sum_remaining_pv_mean_2048': 'Penalty for Remaining PV',
    'custom/reward_mean_2048': 'Average Reward',
    'custom/sum_penalties_os_p_cs_mean_2048': 'Sum of Penalties'
}

# Your subplot setup
n = len(tags_to_plot)
ncols = 2
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5 * nrows))
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])
axes = axes.flatten()


def smooth_data(values, factor=0.9):
    smoothed = np.zeros_like(values)
    for i in range(len(values)):
        if i == 0:
            smoothed[i] = values[i]
        else:
            smoothed[i] = smoothed[i - 1] * factor + values[i] * (1 - factor)
    return smoothed


# Process each log file for the current tag
for idx, (tag, display_name) in enumerate(tags_to_plot.items()):
    steps = []
    values = []

    for log_file in log_files:
        event_acc = EventAccumulator(log_file)
        event_acc.Reload()  # Loads the file

        if tag in event_acc.scalars.Keys():
            scalar_events = event_acc.Scalars(tag)
            steps += [e.step for e in scalar_events]
            values += [e.value for e in scalar_events]

    smoothed_values = smooth_data(values)

    # Your plotting code
    axes[idx].plot(steps, values, alpha=0.2, label='Original')
    axes[idx].plot(steps, smoothed_values, label='Smoothed')
    axes[idx].set_xlabel('Step')
    axes[idx].set_ylabel('Value')
    axes[idx].set_title(display_name)  # Use the display name here
    axes[idx].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[idx].legend(loc='lower right')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'./plots/custom_multiple/{session_prefix}.pdf')
plt.savefig(f'/home/laurenz/Documents/DAI/_Thesis/git/Thesis_template/Figures/plots/tensorboard/custom_multiple/{session_prefix}.pdf')

plt.show()
