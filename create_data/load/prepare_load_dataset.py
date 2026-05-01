import pandas as pd

# Load the time data
time_data = pd.read_csv('HTW2015/time_datevec_MEZ.csv', header=None)
time_data.columns = ['year', 'month', 'day', 'hour', 'minute', 'second']
time_data['datetime'] = pd.to_datetime(time_data[['year', 'month', 'day', 'hour', 'minute', 'second']])

# Change year to 2024 for leap year
time_data['year'] = 2024
time_data['datetime'] = time_data['datetime'].apply(lambda dt: dt.replace(year=2024))

# Initialize an empty DataFrame for the final output
output_df = pd.DataFrame()
output_df['datetime'] = time_data['datetime'].dt.floor('15T').drop_duplicates()  # 15-minute intervals


# Function to load and process power data
def process_power_data(filenames):
    total_power_data = None

    # Sum the power data from all three phases
    for filename in filenames:
        power_data = pd.read_csv(filename, header=None)
        if total_power_data is None:
            total_power_data = power_data
        else:
            total_power_data += power_data  # Summing the power data from each phase

    # Average the total power data over 15-minute intervals
    averaged_data = total_power_data.groupby(total_power_data.index // 15).mean()

    return averaged_data


# Assuming 74 profiles, process each set of PL files for each profile
for profile in range(1, 75):
    filenames = [f'HTW2015/PL{phase}.csv' for phase in range(1, 4)]  # Filenames for all three phases
    averaged_data = process_power_data(filenames)

    # Add the averaged data to the output DataFrame
    output_df[f'profile_{profile}'] = averaged_data.iloc[:, profile - 1].values / 1000  # Adjust index if necessary

# Fill 29th February with the same data as 22nd February for each profile
feb_22_data = output_df[(output_df['datetime'].dt.month == 2) & (output_df['datetime'].dt.day == 22)]
feb_29_data = feb_22_data.copy()
feb_29_data['datetime'] = feb_29_data['datetime'].apply(lambda dt: dt.replace(day=29))
output_df = pd.concat([output_df, feb_29_data])

# Extend data past midnight on 31st December
dec_24_data = output_df[(output_df['datetime'].dt.month == 12) & (output_df['datetime'].dt.day == 24) & (output_df['datetime'].dt.hour < 3)]
jan_1_data = dec_24_data.copy()
jan_1_data['datetime'] = jan_1_data['datetime'].apply(lambda dt: dt.replace(month=1, day=1, year=2025))
output_df = pd.concat([output_df, jan_1_data])

# Remove duplicate rows and sort by datetime
output_df = output_df.drop_duplicates().sort_values(by='datetime').round(5)

# Save the final DataFrame to a CSV file
output_df.to_csv('output_15min_intervals.csv', index=False)
