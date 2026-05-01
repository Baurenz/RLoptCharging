import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV data
def read_data(file_path):
    return pd.read_csv(file_path, parse_dates=['Datetime (UTC)', 'Datetime (Local)'])

# Filter data for a specific date
def filter_data_by_date(data, date):
    return data[data['Datetime (Local)'].dt.date == pd.to_datetime(date).date()]

# Plot data for a specific date
def plot_data_for_date(file_path, date):
    data = read_data(file_path)
    filtered_data = filter_data_by_date(data, date)

    # Convert price from EUR/MWh to EUR/kWh
    filtered_data['Price (EUR/kWh)'] = filtered_data['Price (EUR/MWhe)'] / 1000

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data['Datetime (Local)'], filtered_data['Price (EUR/kWh)'])
    plt.title(f"Energy Prices on {date} (EUR/kWh)")
    plt.xlabel('Time')
    plt.ylabel('Price (EUR/kWh)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

# Usage
file_path = 'Germany.csv'  # Replace with the path to your CSV file
plot_data_for_date(file_path, '2015-01-01')  # Replace with the desired date
