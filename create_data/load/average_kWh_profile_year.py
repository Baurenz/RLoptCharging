import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('load_15min_intervals_74_kW.csv')

# Convert the 'datetime' column to datetime objects
df['datetime'] = pd.to_datetime(df['datetime'])

# Ensure the data covers the specified period
start_date = '2024-01-01 00:00:00'
end_date = '2024-12-31 23:45:00'
df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

# Calculate the total energy consumption for each household
# Since each record represents 15 minutes, we multiply each value by (15/60) to convert kW to kWh
total_consumption_per_profile = df.iloc[:, 1:].multiply(15/60).sum()

# Print the total energy consumption for each household
print("Total energy consumption in kWh for each household:")
print(total_consumption_per_profile)

# Calculate the average energy consumption across all households
average_consumption = total_consumption_per_profile.mean()
print("\nAverage energy consumption across all households:", average_consumption, "kWh")

# Find and print the maximum load value in the dataset
max_load_value = df.iloc[:, 1:].max().max()  # First max() gets the maximum per column, second max() gets the overall maximum
print("\nMaximum load value found in the dataset:", max_load_value, "kW")


# Visualize the distribution of load values for a representative household
# sample_household = df.columns[1]  # Change this to analyze a different household
plt.hist(df.iloc[:, 1:].values.flatten(), bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Load Values Across All Households')
plt.xlabel('Load (kW)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Calculate descriptive statistics
descriptive_stats = df.iloc[:, 1:].describe()
print("\nDescriptive Statistics for Load Values:")
print(descriptive_stats)


# Count the number of load values below 10 kW
# count_below_10 = (df.iloc[:, 1:] < 10).values.sum()
# print("\nNumber of load values below 10 kW:", count_below_10)

total_values_count = df.iloc[:, 1:].size
print("\nTotal number of load values in the dataset:", total_values_count)

# Count the number of load values below 10 kW
count_below_10 = (df.iloc[:, 1:] < 5).sum().sum()  # First sum() over columns, second sum() over rows
print("Number of load values below 5 kW:", count_below_10)



count_above_10 = (df.iloc[:, 1:] > 8).sum()

# Get the 20 profiles with the most values above 10 kW
top_20_profiles_above_10 = count_above_10.nlargest(20)

# Print the profile names and the count of values above 10 kW for the top 20 profiles
print("Top 20 profiles with the most values above 8 kW:")
print(top_20_profiles_above_10)