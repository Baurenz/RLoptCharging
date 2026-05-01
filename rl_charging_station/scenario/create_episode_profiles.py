import json
from datetime import datetime
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates


def pv_profile(time_index, pv_type):
    hours_from_noon = np.abs(time_index.hour - 12 + time_index.minute / 60)
    # Square the Gaussian distribution to make it sharper
    base_curve = np.exp(-0.5 * (hours_from_noon / 3.5) ** 2) ** 2

    if pv_type == "pv1":
        # Add small random noise proportional to the base_curve
        noise = np.random.normal(0, 0.3, len(time_index)) * base_curve
        pv = base_curve + noise
    elif pv_type == "pv2":
        # Add larger random noise proportional to the base_curve
        noise = np.random.normal(0, 0.3, len(time_index)) * base_curve
        pv = base_curve + noise

    # Normalize the profile to [0, 1] range
    pv = (pv - pv.min()) / (pv.max() - pv.min())

    # Scale to [0, 10] range
    pv = pv * 0.015

    # Ensure that the values do not go below 0
    pv = np.maximum(pv, 0)

    return pv


def residential_load_profile(time_index):
    # Create a bell curve centered around 7 AM and 7 PM
    morning_curve = np.exp(-0.5 * ((time_index.hour - 7 + time_index.minute / 60) / 2) ** 2)
    evening_curve = np.exp(-0.5 * ((time_index.hour - 19 + time_index.minute / 60) / 2) ** 2)
    load = morning_curve + evening_curve

    # Add random noise
    noise = np.random.normal(0, 0.05, len(time_index))
    load += noise

    # Ensure that the values do not go below 0
    load = np.maximum(load, 0)

    # Normalize and scale to 20 kW (0.020 MW)
    load = (load / load.max()) * 0.020
    return load


def production_load_profile(time_index):
    # Create a bell curve centered around noon
    load = np.exp(-0.5 * ((time_index.hour - 12 + time_index.minute / 60) / 4) ** 2)

    # Add random noise
    noise = np.random.normal(0, 0.015, len(time_index))
    load += noise

    # Ensure that the values do not go below 0
    load = np.maximum(load, 0)

    # Normalize and scale to 20 kW (0.020 MW)
    load = (load / load.max()) * 0.020
    return load


bus_data = {}  # Replaces plot_data

start_time_create_data = time.time()


def create_new_profiles(profile_json, use_irradiance):
    # Ensure the data directory exists
    os.makedirs('../files/data', exist_ok=True)

    today = datetime.today().strftime('2023-07-21')
    days = 240
    periods = 102
    frequency = '15min'

    total_periods = periods * days
    time_index = pd.date_range(start=today, periods=total_periods, freq=frequency)

    # Load the JSON
    with open(profile_json, "r") as file:
        data = json.load(file)

    buses = data["buses"]
    connections = data["connections"]
    components = data["components"]

    for bus, comps in components.items():
        for comp_key, comp_value in comps.items():
            # Skip components other than "pv" and "load"
            if comp_key not in ["pv", "load"]:
                continue

            profile_name = f"{bus}_{comp_key}"
            profile_file_name = f"dynamic_profiles/20000/{profile_name}.csv"

            for day in range(days):
                # Initialize day_profile to ensure it's always defined
                day_profile = np.array([])

                # Create a new time index for each day starting from midnight
                day_start = pd.to_datetime(today) + pd.Timedelta(days=day)
                day_time_index = pd.date_range(start=day_start, periods=periods, freq=frequency)

                # Generate the profile for the current day
                if comp_key == "pv" and not use_irradiance:
                    pv_profile_type = comp_value["profile"]
                    day_profile = pv_profile(day_time_index, pv_profile_type)
                elif comp_key == "load":
                    if comp_value == "load1":
                        day_profile = residential_load_profile(day_time_index)
                    elif comp_value == "load2":
                        day_profile = production_load_profile(day_time_index)

                # If day_profile is empty, skip to the next iteration
                if day_profile.size == 0:
                    continue

                # Convert to pandas Series
                day_profile_series = pd.Series(data=day_profile, index=day_time_index)
                day_profile_series.name = profile_name

                # Append or write to CSV
                if day == 0:
                    day_profile_series.to_csv(f'../../data/{profile_file_name}', header=True)
                else:
                    day_profile_series.to_csv(f'../../data/{profile_file_name}', mode='a', header=False)

                # Store in bus_data for later use
                if bus not in bus_data:
                    bus_data[bus] = {}
                if profile_name not in bus_data[bus]:
                    bus_data[bus][profile_name] = day_profile_series
                else:
                    bus_data[bus][profile_name] = bus_data[bus][profile_name].append(day_profile_series)

    end_time_create_data = time.time()
    print(f"create_data took {end_time_create_data - start_time_create_data} seconds")



def main(profile_json):
    # Encapsulate the script's main functionality here

    # Call the function to create new profiles
    create_new_profiles(profile_json)

    # Load the components from an existing JSON file for demonstration
    with open(profile_json, "r") as file:
        data = json.load(file)

    components = data["components"]

    # Visualization loop, assuming 'components' is defined and loaded correctly
    for bus, comps in components.items():
        fig, ax = plt.subplots(figsize=(15, 7))

        for comp_key, comp_value in comps.items():
            if comp_key not in ["pv", "load"]:  # Only proceed if the component is 'pv' or 'load'
                continue

            profile_name = f"{bus}_{comp_key}"
            profile_file_name = f"dynamic_profiles/20000/{profile_name}.csv"

            # Read the data from CSV
            profile = pd.read_csv(f'../data/{profile_file_name}', index_col=0, parse_dates=True)

            # Filter for the first five days
            start_date = '2023-09-01'
            filtered_profile = profile[
                (profile.index >= pd.to_datetime(start_date)) & (profile.index < pd.to_datetime(start_date) + pd.Timedelta(days=5))]

            if "pv" in profile_name or "gen" in profile_name:
                ax.plot(filtered_profile.index, filtered_profile.iloc[:, 0], label=profile_name, linestyle="-")
            else:
                ax.plot(filtered_profile.index, filtered_profile.iloc[:, 0], label=profile_name, linestyle="--")

        # Formatting the x-axis
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=90)

        ax.set_title(f"Profiles for {bus}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Power (MW)')
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(f"../data/plots/{bus}.png")
        plt.show()


if __name__ == "__main__":

    profile_json = "../../create_data/profile_json/easy_profile_5bus_no_ess.json"  # Adjust the path as necessary
    main(profile_json)
