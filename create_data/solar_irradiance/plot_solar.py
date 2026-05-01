import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Set the font properties to match LaTeX document
plt.rcParams['font.family'] = 'sans-serif'  # Use sans-serif font family
plt.rcParams['font.sans-serif'] = 'Helvetica'  # Specifically, use Helvetica if available

plt.rcParams['mathtext.fontset'] = 'custom'  # Customize math font
plt.rcParams['mathtext.rm'] = 'Palatino'  # Set the default math font to Palatino (or a similar serif font)
plt.rcParams['mathtext.it'] = 'Palatino:italic'  # Italic font for math
plt.rcParams['mathtext.bf'] = 'Palatino:bold'  # Bold font for math

# Load the data
df = pd.read_csv('weather_data_reilingen.csv')

# Convert the 'period_end' column to datetime
df['period_end'] = pd.to_datetime(df['period_end']).dt.tz_localize(None)  # Convert to naive datetime if originally timezone-aware

# Function to plot data for a given date and the next day until 2 AM
def plot_data_for_date(date):
    # Define the start and end of the period to include the next day until 2 AM
    start_period = pd.to_datetime(date)
    end_period = start_period + pd.Timedelta(days=1, hours=2)

    # Filter data for the selected period
    selected_date_data = df[(df['period_end'] >= start_period) & (df['period_end'] < end_period)]

    # Plot settings
    plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

    # Plot for 'cloud_opacity'
    fig, ax = plt.subplots()
    ax.plot(selected_date_data['period_end'], selected_date_data['cloud_opacity'], marker='o', linestyle='-', color='blue')
    ax.set_title('Cloud Opacity on ' + date)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cloud Opacity | %')
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot for 'ghi' and 'gti'
    fig, ax = plt.subplots()
    ax.plot(selected_date_data['period_end'], selected_date_data['ghi'], marker='o', linestyle='-', color='red', label='GHI')
    ax.plot(selected_date_data['period_end'], selected_date_data['gti'], marker='x', linestyle='-', color='green', label='GTI')
    ax.set_title('GHI vs GTI in Reilingen (49.293544 | 8.563727) on ' + date)
    ax.set_xlabel('Time')
    ax.set_ylabel('Irradiance | W/m2')
    ax.legend(loc='best')
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage for a specific date
plot_data_for_date('2023-06-18')
