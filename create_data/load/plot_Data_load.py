import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Assuming plot_style module is already defined as before
from plots_thesis import plot_style


# Function to load data and convert to kW
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    # Convert from W to kW
    for col in df.columns[1:]:
        df[col] = df[col] / 1000
    return df


# Function to plot data for specified dates and household
def plot_load_profiles(df, dates, household, colors):
    plot_style.set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Use a fixed, arbitrary date for plotting (e.g., 2000-01-01)
    arbitrary_date = pd.Timestamp('2000-01-01')

    # Assuming first date is summer and second is winter for simplicity
    date_labels = ['Summer Day', 'Winter Day']

    for date, color, label in zip(dates, colors, date_labels):
        day_data = df[df['datetime'].dt.date == pd.to_datetime(date).date()]
        # Calculate the timedelta and add it to the arbitrary date
        times = [arbitrary_date + (dt - dt.normalize()) for dt in day_data['datetime']]
        ax.plot(times, day_data[household], label=label, color=color)

    # Formatting the x-axis to show time in HH:MM format and set limits from 00:00 to 24:00
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlim([arbitrary_date, arbitrary_date + pd.Timedelta(days=1)])

    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Load (kW)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


# Main function to load data and plot
def main():
    filepath = 'load_15min_intervals_74_kW.csv'
    df = load_data(filepath)
    dates = ['2010-06-23', '2010-12-21']
    household = 'profile_17'  # Specify the household to plot

    # Define colors for summer and winter plots
    colors = ['#9881FB', '#59CC40']  # Adjust the colors as per your preference

    plot_load_profiles(df, dates, household, colors)


if __name__ == '__main__':
    main()
