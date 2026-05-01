import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def get_irr_temp_cloud_raw():
    # Define the file path to the solar irradiance CSV file
    file_path = 'data/solar_irradiance/solar_temp_cloud_010115-301123_reilingen.csv'

    # Load the GTI and period_end data from the CSV file
    ambient_data_raw = pd.read_csv(file_path, usecols=['Datetime (Local)', 'gti', 'air_temp', 'cloud_opacity'])
    # Convert the 'period_end' column to datetime, ensuring correct time zone handling
    ambient_data_raw['Datetime (Local)'] = pd.to_datetime(ambient_data_raw['Datetime (Local)']).dt.tz_localize(None)
    irr_temp_cloud_raw = ambient_data_raw[['Datetime (Local)', 'gti', 'air_temp', 'cloud_opacity']]

    return irr_temp_cloud_raw


def get_day_ahead_price_raw():
    # Load the data from the CSV file
    file_path = 'data/market/Germany_day_ahead.csv'

    day_ahead_prices_raw = pd.read_csv(file_path, usecols=['Datetime (Local)', 'Price (EUR/MWhe)'])

    day_ahead_prices_raw['Price (EUR/kWh)'] = day_ahead_prices_raw['Price (EUR/MWhe)'] / 1000

    # Drop the original price column and rename the new column
    day_ahead_prices_raw.drop('Price (EUR/MWhe)', axis=1, inplace=True)

    return day_ahead_prices_raw


def get_real_load_raw():
    # Define the file path to the load CSV file
    file_path = 'data/load/load_15min_intervals_74_kW.csv'

    # Load all the data from the CSV file, assuming the first column is 'datetime' and the rest are load profiles
    real_load_raw = pd.read_csv(file_path)
    profiles_to_drop = ['profile_70', 'profile_8', 'profile_9', 'profile_35']
    real_load_raw = real_load_raw.drop(columns=profiles_to_drop)

    # Convert the 'datetime' column to datetime objects, ensuring correct time zone handling if necessary
    real_load_raw['datetime'] = pd.to_datetime(real_load_raw['datetime'])

    return real_load_raw


# def get_next_episode_bus_data(bus_dict, day_count, irradiance_flag, solar_irradiance_temp_cloudopacity_episode, real_load_flag,
#                               real_load_episode):
#     """for each new episode new data has to be loaded"""
# 
#     for bus_id in bus_dict:
#         if bus_dict[bus_id].has_pv:
#             if irradiance_flag:
#                 # If irradiance_flag is True, use the irradiance data method
#                 bus_dict[bus_id].get_pv_prod_ep_ambient(solar_irradiance_temp_cloudopacity_episode)
#             else:
#                 # If irradiance_flag is False, uses (file based) method
#                 bus_dict[bus_id].get_pv_production_episode(day_count, bus_id)
# 
#         if bus_dict[bus_id].has_load:
#             if real_load_flag:
#                 bus_dict[bus_id].get_real_load_episode(real_load_episode)
#             else:
#                 bus_dict[bus_id].get_load_episode(day_count)
# 
#     return bus_dict


def get_ep_bus_data_eval(bus_dict, results_data_dict):
    """for each new episode new data has to be loaded"""
    # TODO: maybe EOL. not sure if i use it

    for bus_id in bus_dict:
        if bus_dict[bus_id].has_pv:
            # Get the PV production data from results_data_dict using bus_id
            bus_dict[bus_id].pv.pv_prod_episode = results_data_dict['renewable'][str(bus_id)]

        if bus_dict[bus_id].has_load:
            # Get the load data from results_data_dict using bus_id
            bus_dict[bus_id].load_episode = results_data_dict['load'][str(bus_id)]

    return bus_dict


def get_random_date_ep():
    # Define the date range
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2023, 11, 29)

    # Generate a random date within the range
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))

    return random_date


def get_eval_date_ep(ep_count):
    # Define the start date
    start_date = datetime(2019, 1, 1)

    # Increment the start date by ep_count days
    ep_date = start_date + timedelta(days=ep_count)

    return ep_date
def get_cyclical_day_of_year(date):
    day_of_year = date.timetuple().tm_yday
    days_in_year = 365
    sin_day = (np.sin(2 * np.pi * day_of_year / days_in_year) + 1) / 2
    cos_day = (np.cos(2 * np.pi * day_of_year / days_in_year) + 1) / 2
    return [sin_day, cos_day]


def get_ep_price_data(da_price_raw, random_date_ep):
    # Convert the 'Datetime (UTC)' column to datetime format
    da_price_raw['Datetime (Local)'] = pd.to_datetime(da_price_raw['Datetime (Local)'])

    # Calculate the end datetime (next day + 2 hours)
    end_datetime = random_date_ep + timedelta(days=1, hours=2, minutes=45)

    # Filter the DataFrame for the period from the random date to the end datetime
    day_ahead_price_episode = da_price_raw[
        (da_price_raw['Datetime (Local)'] >= random_date_ep) &
        (da_price_raw['Datetime (Local)'] < end_datetime)]

    day_ahead_price_episode.set_index('Datetime (Local)', inplace=True)
    # TODO: do i want to apply this?
    #########################################################################################################
    # apply conditions from Flexible day-ahead based awattar tariff: (https://www.awattar.de/tariffs/hourly)
    # day_ahead_price_episode['Price (EUR/kWh)'] = day_ahead_price_episode['Price (EUR/kWh)'] * 1.03 + 0.1671
    # # Apply the bounds of -0.80 €/kWh and 0.80 €/kWh
    # day_ahead_price_episode['Price (EUR/kWh)'] = day_ahead_price_episode['Price (EUR/kWh)'].clip(lower=-0.80, upper=0.80)

    da_price_ep_exp = day_ahead_price_episode.resample('15T').ffill()
    da_price_ep_exp.reset_index(inplace=True)
    da_price_ep_exp = da_price_ep_exp['Price (EUR/kWh)'].values
    da_price_ep_norm, da_range_ep = get_norm_day_ahead_price(da_price_ep_exp)

    return da_price_ep_exp, da_price_ep_norm, da_range_ep


def get_ep_irr_temp_cloud(irr_temp_cloud_raw, random_date_episode):
    # Ensure the date column is in datetime format
    irr_temp_cloud_raw['Datetime (Local)'] = pd.to_datetime(irr_temp_cloud_raw['Datetime (Local)'])
    # Calculate the end datetime (next day + 2 hours)
    end_datetime = random_date_episode + timedelta(days=1, hours=2, minutes=45)

    # Filter the DataFrame for the period from the random date to the end datetime
    episode_data = irr_temp_cloud_raw[
        (irr_temp_cloud_raw['Datetime (Local)'] >= random_date_episode) &
        (irr_temp_cloud_raw['Datetime (Local)'] < end_datetime)]

    episode_data.set_index('Datetime (Local)', inplace=True)

    # Extract irradiance and temperature values
    irradiance_values = episode_data['gti'].values
    temp_values = episode_data['air_temp'].values
    cloud_opacity_values = episode_data['cloud_opacity']

    # Stack irradiance and temperature values horizontally
    # so that each temperature value is right next to its corresponding irradiance value
    irr_temp_cloud_ep = np.column_stack((irradiance_values, temp_values, cloud_opacity_values))

    return irr_temp_cloud_ep


def get_ep_load_data(real_load_raw, random_date_episode):
    # Ensure the 'datetime' column is in datetime format
    real_load_raw['datetime'] = pd.to_datetime(real_load_raw['datetime'])

    # Adjust the year of the random_date_episode to 2024 (it's a leap year, so it has all dates available)
    # (the og load data is only available in 2010)
    adjusted_date = random_date_episode.replace(year=2024)

    # Calculate the end datetime (next day + 2 hours and 45 minutes), adjusting the year to 2010
    end_datetime = adjusted_date + timedelta(days=1, hours=2, minutes=45)

    # Filter the DataFrame for the period from the adjusted random date to the end datetime
    episode_data = real_load_raw[
        (real_load_raw['datetime'] >= adjusted_date) &
        (real_load_raw['datetime'] < end_datetime)]

    episode_load_values = episode_data.drop(columns=['datetime']).values

    return episode_load_values


def init_cs_pres_soc(instance, ep_count):
    rng = np.random.RandomState(ep_count)
    arrival_probabilities = [(0, 0.85), (8, 0.1), (32, 0.05), (48, 0.1), (64, 0.15), (80, 0.15), (96, 0.15)]
    departure_probabilities = [(0, 0.04), (28, 0.2), (36, 0.1), (48, 0.15), (64, 0.1), (80, 0.1), (96, 0.1)]

    n_cs = instance.n_cs
    soc_cs_init = - np.ones([n_cs, instance.last_timestep + 1])
    present_cars = np.zeros([n_cs, instance.last_timestep + 1])
    arrival_t = []
    departure_t = []

    def get_probability(probabilities, timestep):
        for time, prob in reversed(probabilities):
            if timestep >= time:
                return prob
        return 0

    for car in range(n_cs):
        present = False
        arrival_car = []
        departure_car = []
        arrival_hour = None  # Variable to track the arrival timestep

        for timestep in range(instance.last_timestep):
            if not present and timestep < instance.last_timestep - 2:  # Preventing arrivals in the last two timesteps
                arrival_prob = get_probability(arrival_probabilities, timestep)
                present = rng.rand() < arrival_prob
                if present:
                    soc_cs_init[car, timestep] = rng.randint(10, 61) / 100
                    arrival_car.append(timestep)
                    arrival_hour = timestep  # Set the arrival timestep when the car arrives

            elif present:
                # Check for departure only if at least 6 timesteps have passed since arrival
                if timestep >= arrival_hour + 14:
                    departure_prob = get_probability(departure_probabilities, timestep)
                    will_leave = rng.rand() < departure_prob
                    if will_leave:
                        departure_car.append(timestep)
                        present = False
                        arrival_hour = None  # Reset the arrival timestep

            present_cars[car, timestep] = 1 if present else 0

        # Ensure departure at the last timestep if still present
        if present:
            # TODO maybe thats not right? maybe it should be present but leave?
            present_cars[car, instance.last_timestep] = 0
            departure_car.append(instance.last_timestep)

        arrival_t.append(arrival_car)
        departure_t.append(departure_car)

    evolution_of_cars = np.sum(present_cars, axis=0)

    return soc_cs_init, arrival_t, departure_t, evolution_of_cars, present_cars


def reset_init_ess_soc(instance, ep_count):
    # Ensure that the limits are between 0 and 1
    rng = np.random.RandomState(ep_count)
    n_ess = instance.n_ess

    lower_limit = 0.2
    upper_limit = 0.8

    # Initialize soc_ess with zeros for all timesteps
    soc_ess = np.zeros([n_ess, instance.last_timestep + 1])

    # Generate random SoC values for the first timestep only
    soc_ess[:, 0] = rng.uniform(lower_limit, upper_limit, n_ess)

    return soc_ess


def eval_init_cs_pres_soc(instance, n_cs, initial_data_dict):
    # Extract data from the dictionary
    soc_cs_init = np.array(initial_data_dict['soc_cs'])  # Assuming 'boc' contains the initial state of charge for each charging station
    arrival_t = initial_data_dict['arrival_t']
    departure_t = initial_data_dict['departure_t']
    evolution_of_cars = np.array(initial_data_dict['evolution_of_cars'])
    present_cars = np.array(initial_data_dict['present_cars'])

    # Ensure the data shapes are correct, especially if the number of charging stations (n_cs) is different from what was originally saved
    if soc_cs_init.shape[0] != n_cs:
        raise ValueError(
            f"The provided initial data has a different number of charging stations ({soc_cs_init.shape[0]}) than expected ({n_cs}).")

    # No need to generate data, as we are using the provided initial_data_dict
    # Return the data in the same format as reset_init_cs_presence_soc
    return soc_cs_init, arrival_t, departure_t, evolution_of_cars, present_cars


def eval_init_ess_soc(instance, n_ess, initial_data_dict):
    # Initialize soc_ess with zeros for all timesteps
    soc_ess = np.zeros([n_ess, instance.last_timestep + 1])

    # Load the initial SoC values from the dictionary for the first timestep
    # and ensure it's a numpy array to facilitate indexing
    initial_soc = np.array(initial_data_dict['soc_ess'])

    # Ensure that the provided initial SoC data matches the expected number of ESS units
    if initial_soc.shape[0] != n_ess:
        raise ValueError(
            f"The provided initial data has a different number of ESS units ({initial_soc.shape[0]}) than expected ({n_ess}).")

    # Fill the first timestep of soc_ess with the loaded initial SoC values
    soc_ess[:, 0] = initial_soc[:, 0]

    # The rest of soc_ess is already initialized to zeros
    return soc_ess


def get_norm_day_ahead_price(da_price_ep):
    # TODO: not sure how to normalize energy prices. min max approach or only max approach?
    min_price = np.min(da_price_ep)
    max_price = np.max(da_price_ep)

    if min_price <= 0:
        print(min_price)
    # Step 3: Normalize the prices
    # Avoid division by zero in case all prices in the episode are the same
    if max_price != min_price:
        da_price_ep_norm = (da_price_ep - min_price) / (max_price - min_price)
    else:
        # Handle the case where all prices are the same (e.g., by setting all normalized prices to the same value, such as 0.5)
        da_price_ep_norm = np.full_like(da_price_ep, 0.5)

    da_range_ep = max_price - min_price  # will be needed to correctly incorporate different byuing and selling prices within reward calculation

    return da_price_ep_norm, da_range_ep
