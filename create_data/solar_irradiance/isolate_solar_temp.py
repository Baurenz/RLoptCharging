import pandas as pd

# Load the data
df = pd.read_csv('weather_data_reilingen.csv')

# Convert the 'period_end' column to datetime
df['period_end'] = pd.to_datetime(df['period_end'])

# Select only the 'ghi' and 'period_end' columns
datetime_gti_airtemp = df[['period_end', 'gti', 'air_temp', 'cloud_opacity']]

# Save the filtered data to a new CSV file
datetime_gti_airtemp.to_csv('solar_temp_cloud_010115-301123_reilingen.csv', index=False)

