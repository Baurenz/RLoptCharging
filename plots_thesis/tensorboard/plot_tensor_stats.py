import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import matplotlib.ticker as ticker

from plots_thesis import plot_style

plot_style.set_plot_style()
plt.rcParams['figure.figsize'] = (7, 5)

plt.rcParams['font.size'] = 18.5

# Path to your logs directory
model_type = 'DDPG'
session_prefix = 'scenarioutop'
session_id = 1

log_base = '../../solvers/results/logs/'
model_dir = f'{model_type}/{session_prefix}{session_id}/DDPG_0'
log_dir = log_base + model_dir

# List all .tfevents files in the log directory and subdirectories
log_files = glob.glob(os.path.join(log_dir, '**', '*.tfevents.*'), recursive=True)

# Sort files by creation time
log_files.sort(key=os.path.getmtime)

# Define your tags and their labels here
tags = {
    'custom/03_reward_mean_4032': r'$r(S_t, A_t)$',
    'custom/wn_selling_ext_grid_mean_4032': r'$r_{\mathrm{grid}}$',
    'custom/wn_buying_ext_grid_mean_4032': r'$p_{\mathrm{grid}}$',
    # 'custom/penalty_sum_remaining_pv_mean_4032': r'$p_{\mathrm{pv}}$',
    'custom/penalty_ev_mean_4032': r'$ p_{\mathrm{cs}}$',
    # 'custom/penalty_lines_total_mean_4032': r'$ p_{\mathrm{line}}$'
    # Add more tags and labels as needed
}

# 'custom/03_reward_mean_4032': r'$r(S_t, A_t)$',
# 'custom/wn_buying_ext_grid_mean_4032': r'$p_{\text{grid}}$',
# 'custom/penalty_sum_remaining_pv_mean_4032': r'$p_{\text{pv}}$',
# 'custom/penalty_ev_mean_4032': r'$ p_{\text{cs}}$'

# Dictionary to hold aggregated data for each tag
data = {tag: {'steps': [], 'values': []} for tag in tags.keys()}

# Process each log file
for log_file in log_files:
    for e in tf.compat.v1.train.summary_iterator(log_file):
        for v in e.summary.value:
            if v.tag in tags:
                data[v.tag]['steps'].append(e.step)
                data[v.tag]['values'].append(v.simple_value)


def smooth_data(values, factor=0.9):
    smoothed = np.zeros_like(values)
    for i in range(len(values)):
        if i == 0:
            smoothed[i] = values[i]
        else:
            smoothed[i] = smoothed[i - 1] * factor + values[i] * (1 - factor)
    return smoothed


# Plotting
for index, (tag, metrics) in enumerate(data.items()):
    # Apply smoothing to the values
    smoothed_values = smooth_data(metrics['values'], factor=0.9)
    color = plot_style.color_cycle[index]

    # Plot the original data (faded)
    plt.plot(metrics['steps'], metrics['values'], alpha=0.3,
             color=color)  # label=f'{tags[tag]} Original'  # Faded plot of the original data

    # Plot the smoothed data
    # Check if the current tag is 'Reward'
    if index == 0:
        # Plot the smoothed data with a thicker line
        plt.plot(metrics['steps'], smoothed_values, label=f'{tags[tag]} Smoothed', color=color, linewidth=3)
    else:
        # Plot the smoothed data with normal line width
        plt.plot(metrics['steps'], smoothed_values, label=f'{tags[tag]} Smoothed', color=color)

# Set labels and title

plt.xlabel('Step')
plt.ylabel('Value')
# plt.title('Metrics with Smoothing')

plt.xlim((-5000, 3.5e5))

plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

# Set the y-axis limits if necessary
# plt.ylim((-11, 3))

# Enable grid for every tick
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show legend
plt.legend(fontsize='small', loc='lower right')

plt.tight_layout()
file_name = f'tensorboard/mlt_mtx_{model_type}_{session_prefix}_{session_id}comp'
plot_style.save_fig(file_name)
plt.show()
