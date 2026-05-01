import os
import json
import numpy as np
import matplotlib.pyplot as plt


def evaluate_soc(selected_model, model, session_id):
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    # Directory where the JSON files are stored
    results_dir = f'solvers/evaluation/Results/{selected_model}_{model}{session_id}'
    files = [f for f in os.listdir(results_dir) if f.startswith('results_') and f.endswith('.json')]

    # Data structure to hold the SoC values for each car
    car_soc_values = {}

    for file in files:
        with open(os.path.join(results_dir, file), 'r') as f:
            data = json.load(f)
            cs_soc_leave = data.get('cs_soc_leave', [])

            for car_index, soc_values in enumerate(cs_soc_leave):
                if car_index not in car_soc_values:
                    car_soc_values[car_index] = []
                car_soc_values[car_index].extend(soc_values)

    # Convert the SoC values into a format suitable for violin plots
    soc_values_for_plot = [values for values in car_soc_values.values()]

    # Plotting
    plt.figure(figsize=(10, 6))
    parts = plt.violinplot(soc_values_for_plot, showmeans=False, showmedians=True)

    # Customizing the violin plot's appearance
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    plt.xticks(range(1, len(soc_values_for_plot) + 1), [f'Car {i}' for i in range(len(soc_values_for_plot))])
    plt.xlabel('Car Index')
    plt.ylabel('SoC at Departure')
    plt.title('Distribution of SoC at Departure for Each Car')
    plt.figure()
    plt.show()


if __name__ == '__main__':

    session_id = 12
    evaluate_soc(session_id)
