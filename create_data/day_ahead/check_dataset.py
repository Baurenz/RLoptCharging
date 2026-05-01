import pandas as pd

# Assuming the file 'Germany.csv' is in the same directory as the script
filename = 'Germany.csv'

try:
    # Load the CSV file
    data = pd.read_csv(filename)

    # Extract the 'Datetime (Local)' column
    datetime_local = pd.to_datetime(data['Datetime (Local)'])

    # Group by date and count the number of entries per date
    counts = datetime_local.dt.date.value_counts().sort_index()

    # Find dates with missing entries (less than 24 hours)
    missing_entries = counts[counts < 24]

    # For each date with missing entries, find which hours are missing
    missing_hours = {}
    for date in missing_entries.index:
        expected_hours = set(range(24))
        actual_hours = set(datetime_local[datetime_local.dt.date == date].dt.hour)
        missing_hours[date] = expected_hours - actual_hours

    missing_hours
except FileNotFoundError:
    print(f"File '{filename}' not found in the directory.")
except Exception as e:
    print(f"An error occurred: {e}")