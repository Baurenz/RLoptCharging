import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime

# Function to generate random dates
def random_dates(start, end, n=10):
    start_u = start.toordinal()
    end_u = end.toordinal()
    return [datetime.fromordinal(random.randint(start_u, end_u)).date() for _ in range(n)]

# Read the CSV data
def read_data(file_path):
    return pd.read_csv(file_path, parse_dates=['Datetime (UTC)', 'Datetime (Local)'])

# Filter data for a specific date
def filter_data_by_date(data, date):
    return data[data['Datetime (Local)'].dt.date == pd.to_datetime(date).date()]

# Plot data for multiple random dates
def plot_data_for_random_dates(file_path, start_date, end_date):
    data = read_data(file_path)
    random_days = random_dates(start_date, end_date)

    plt.figure(figsize=(12, 8))

    # Use the string representation of time component of one day for x-axis
    sample_day = filter_data_by_date(data, random_days[0])
    x_times = sample_day['Datetime (Local)'].dt.time.apply(lambda t: t.strftime('%H:%M'))

    for day in random_days:
        filtered_data = filter_data_by_date(data, day)
        # Convert price from EUR/MWh to EUR/kWh
        filtered_data['Price (EUR/kWh)'] = filtered_data['Price (EUR/MWhe)'] / 1000
        plt.plot(x_times, filtered_data['Price (EUR/kWh)'], label=day.strftime('%Y-%m-%d'))

    plt.title("Energy Prices for Random Days (Overlaid by Time of Day)")
    plt.xlabel('Time of Day')
    plt.ylabel('Price (EUR/kWh)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

# Usage
file_path = 'Germany.csv'  # Replace with the path to your CSV file
plot_data_for_random_dates(file_path, datetime(2015, 1, 1), datetime(2023, 11, 30))
