import csv
import requests
from datetime import datetime, time, timedelta
import os

# flag to create 96 timsteps (15 minute intervals) -- for now same price for subsequent 3 timesteps

create_96 = True

dirpath = '../data/market/'
if not os.path.exists(dirpath):
    os.makedirs(dirpath)


def convert_timestamp(timestamp_in_millis):
    return datetime.fromtimestamp(timestamp_in_millis / 1000).strftime('%Y-%m-%d %H:%M:%S')


today = datetime.combine(datetime.today(), time.min)
today_shifted = today - timedelta(hours=6)
timestamp_today_shifted = int(today_shifted.timestamp() * 1000)  # convert to milliseconds
timestamp_today = int(today.timestamp() * 1000)  # convert to milliseconds

end_time = today_shifted + timedelta(hours=31)
timestamp_end = int(end_time.timestamp() * 1000)

response = requests.get(f"https://api.awattar.de/v1/marketdata?start={timestamp_today}&end={timestamp_end}")
data = response.json()["data"]

today_str = today.strftime('%Y-%m-%d')  # get current date as a string in 'YYYY-MM-DD' format
filepath = f'../data/market/{today_str}.csv'  # construct the file path

with open(filepath, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["start_timestamp", "marketprice", "unit"])  # write header

    for item in data:
        original_timestamp = item["start_timestamp"]
        for _ in range(4 if create_96 else 1):
            writer.writerow([convert_timestamp(original_timestamp), max(0, item["marketprice"]) / 1000, 'Eur/kWh'])
            original_timestamp += 15 * 60 * 1000  # increment by 15 minutes in milliseconds
