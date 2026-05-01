import json
from datetime import datetime
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



profile_json = "profile_json/easy_profile_5bus.json"

# Ensure the data directory exists
os.makedirs('../data', exist_ok=True)

today = datetime.today().strftime('2023-07-21')
periods = 102
frequency = '15min'

time_index = pd.date_range(start=today, periods=periods, freq=frequency)

# Load the JSON
with open(profile_json, "r") as file:
    data = json.load(file)

buses = data["buses"]
connections = data["connections"]
components = data["components"]


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


def v2g_profile(time_index, v2g_type):
    v2g = np.zeros_like(time_index.hour, dtype=float)

    if v2g_type == "v2g1":
        v2g[(time_index.hour >= 20) | (time_index.hour < 6)] = 0.011
        v2g[((time_index.hour >= 6) & (time_index.hour <= 9)) |
            ((time_index.hour >= 17) & (time_index.hour <= 20))] = -0.01
    elif v2g_type == "v2g2":
        v2g[(time_index.hour >= 21) | (time_index.hour < 5)] = 0.011
        v2g[((time_index.hour >= 7) & (time_index.hour <= 10)) |
            ((time_index.hour >= 18) & (time_index.hour <= 21))] = 0.01

    return v2g


bus_data = {}  # Replaces plot_data

start_time_create_data = time.time()

for bus, comps in components.items():
    for comp_key, comp_value in comps.items():
        profile_name = f"{bus}_{comp_key}"
        profile_file_name = f"dynamic_profiles/{profile_name}.csv"

        if comp_key == "pv":
            profile = pd.Series(data=pv_profile(time_index, comp_value), index=time_index)
            profile.name = profile_name
            profile.to_csv(f'../data/{profile_file_name}', header=True)

            if bus not in bus_data:
                bus_data[bus] = {}
            bus_data[bus][profile_name] = profile
        elif comp_key == "load":
            if comp_value == "load1":
                profile = pd.Series(data=residential_load_profile(time_index), index=time_index)
            elif comp_value == "load2":
                profile = pd.Series(data=production_load_profile(time_index), index=time_index)
            profile.name = profile_name
            profile.to_csv(f'../data/{profile_file_name}', header=True)
            if bus not in bus_data:
                bus_data[bus] = {}
            bus_data[bus][profile_name] = profile
        elif comp_key == "gen":
            profile = pd.Series(data=np.ones(periods) * (0.01 if comp_value == "gen1" else 0.02), index=time_index)
            profile.name = profile_name
            profile.to_csv(f'../data/{profile_file_name}', header=True)
            if bus not in bus_data:
                bus_data[bus] = {}
            bus_data[bus][profile_name] = profile
        elif comp_key == "v2g":
            profile = pd.Series(data=v2g_profile(time_index, comp_value), index=time_index)
            profile.name = profile_name
            profile.to_csv(f'../data/{profile_file_name}', header=True)
            if bus not in bus_data:
                bus_data[bus] = {}
            bus_data[bus][profile_name] = profile
folder_path = Path('../data/market')
filepath_market = max(folder_path.glob('*.csv'), key=lambda x: x.stat().st_mtime)

end_time_create_data = time.time()
print(f"create_data took {end_time_create_data - start_time_create_data} seconds")


# Reading the market prices
market_data = pd.read_csv(filepath_market, index_col=0, parse_dates=True)
market_prices_series = market_data.iloc[:, 0]  # Assuming prices are in the first column

# Plotting market prices directly
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(market_prices_series.index, market_prices_series, label='Market Price', linestyle="-")
ax.set_title(f"Market Price Profile")
ax.set_xlabel('Time')
ax.set_ylabel('Price ($)')
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(f"../data/plots/market_price.png")
plt.show()

# Plotting data grouped by bus
for bus, profiles in bus_data.items():
    fig, ax = plt.subplots(figsize=(15, 7))

    for profile_name, profile in profiles.items():
        if "pv" in profile_name or "gen" in profile_name:
            ax.plot(profile.index, profile, label=profile_name, linestyle="-")  # Using solid line for production
        else:
            ax.plot(profile.index, profile, label=profile_name, linestyle="--")  # Using dashed line for consumption

    ax.set_title(f"Profiles for {bus}")
    ax.set_xlabel('Time')
    ax.set_ylabel('Power (MW)')
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(f"../data/plots/{bus}.png")  # Saving the figure as a PNG file
    plt.show()
