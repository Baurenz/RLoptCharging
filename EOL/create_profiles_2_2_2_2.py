from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

# Ensure the data directory exists
os.makedirs('../data', exist_ok=True)

today = datetime.today().strftime('2023-07-21')  # get current date as a string in 'YYYY-MM-DD' format
periods = 96
frequency = '15min'

# Create a time index
time_index = pd.date_range(start=today, periods=periods, freq=frequency)


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


pv1 = pd.Series(data=pv_profile(time_index, "pv1"), index=time_index)
pv1.name = "PV1"
pv1.to_csv('../data/pv1.csv', header=True)

pv2 = pd.Series(data=pv_profile(time_index, "pv2"), index=time_index)
pv2.name = "PV2"
pv2.to_csv('../data/pv2.csv', header=True)


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

    return load


def production_load_profile(time_index):
    # Create a bell curve centered around noon
    load = np.exp(-0.5 * ((time_index.hour - 12 + time_index.minute / 60) / 4) ** 2)

    # Add random noise
    noise = np.random.normal(0, 0.015, len(time_index))
    load += noise

    # Ensure that the values do not go below 0
    load = np.maximum(load, 0)

    return load


consumption1 = pd.Series(data=residential_load_profile(time_index) * 0.015, index=time_index)  # in MW
consumption1.name = "Load1"
consumption1.to_csv('../data/consumption1.csv', header=True)

consumption2 = pd.Series(data=production_load_profile(time_index) * 0.010, index=time_index)  # in MW
consumption2.name = "Load2"
consumption2.to_csv('../data/consumption2.csv', header=True)

# Create generator profiles - assuming a flat output
gen1 = pd.Series(data=np.ones(periods) * 0.01, index=time_index)  # constant generation in MW
gen1.name = "Gen1"
gen1.to_csv('../data/gen1.csv', header=True)

gen2 = pd.Series(data=np.ones(periods) * 0.02, index=time_index)  # constant generation in MW
gen2.name = "Gen2"

gen2.to_csv('../data/gen2.csv', header=True)

folder_path = Path('../data/market')

# Get the latest .csv file based on modification time
filepath_market = max(folder_path.glob('*.csv'), key=lambda x: x.stat().st_mtime)

# load the market prices
market_prices = pd.read_csv(filepath_market, index_col=0, parse_dates=True)


# Create v2g profiles - assuming cars are charging from 9 PM to 6 AM (off-peak hours) and discharging from 6-9 AM
# and 5-8 PM (peak hours)


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


v2g1 = pd.Series(data=v2g_profile(time_index, "v2g1"), index=time_index)
v2g1.name = "V2G1"
v2g1.to_csv('../data/v2g1.csv', header=True)

v2g2 = pd.Series(data=v2g_profile(time_index, "v2g2"), index=time_index)
v2g2.name = "V2G2"
v2g2.to_csv('../data/v2g2.csv', header=True)

# Create a figure and set of subplots
fig, axs = plt.subplots(5, 2, figsize=(15, 15))

# Plot each time series
axs[0, 0].plot(time_index, pv1, label="pv1")
axs[0, 0].set_title('pv1')
axs[0, 1].plot(time_index, pv2, label="pv2")
axs[0, 1].set_title('pv2')

axs[1, 0].plot(time_index, consumption1, label="consumption1")
axs[1, 0].set_title('consumption1')
axs[1, 1].plot(time_index, consumption2, label="consumption2")
axs[1, 1].set_title('consumption2')

axs[2, 0].plot(time_index, gen1, label="gen1")
axs[2, 0].set_title('gen1')
axs[2, 1].plot(time_index, gen2, label="gen2")
axs[2, 1].set_title('gen2')

axs[3, 0].plot(time_index, v2g1, label="v2g1")
axs[3, 0].set_title('v2g1')
axs[3, 1].plot(time_index, v2g2, label="v2g2")
axs[3, 1].set_title('v2g2')

# plot the market prices
axs[4, 0].plot(market_prices.index, market_prices['marketprice'], label='Market Price')
axs[4, 0].set_title('Market Price')
axs[4, 0].set_xlabel('Time')
axs[4, 0].set_ylabel('Price (Eur/MWh)')

# Add labels and title
for ax in axs.flat:
    ax.set(xlabel='Time', ylabel='Power (MW)')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.show()
