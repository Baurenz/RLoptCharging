################################################
# Plot to show different load profiles.
###############################################

# TODO: do i need minor grid on y-axis?
# TODO: change name of profiles to: 'Profile 17'
# TODO: legend in the same corner ofc!

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
    return df


# Function to plot data for specified dates and household
def plot_load_profiles(df, dates_per_profile, profiles, colors):
    plot_style.set_plot_style()

    plt.rcParams['font.size'] = 16  # Adjust based on your document's specific needs

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()  # Flatten the 2x2 grid to easily loop through

    # Use a fixed, arbitrary date for plotting (e.g., 2000-01-01)
    arbitrary_date = pd.Timestamp('2000-01-01')

    date_labels = ['Summer Day', 'Winter Day']

    # Calculate the maximum load among the selected dates for consistent y-axis limits
    max_load = 0
    for profile, dates in zip(profiles, dates_per_profile):
        for date in dates:
            day_max = df[df['datetime'].dt.date == pd.to_datetime(date).date()][profile].max()
            max_load = max(max_load, day_max)

    for i, (ax, profile, dates) in enumerate(zip(axs, profiles, dates_per_profile)):
        for date, color, label in zip(dates, colors, date_labels):
            day_data = df[df['datetime'].dt.date == pd.to_datetime(date).date()]
            times = [arbitrary_date + (dt - dt.normalize()) for dt in day_data['datetime']]
            ax.plot(times, day_data[profile], label=label, color=color)

        # Rename the profile for the title
        formatted_profile_name = profile.replace('_', ' ').capitalize()
        ax.set_title(formatted_profile_name, loc='center')

        ax.set_ylim(0, max_load)  # Set consistent y-axis limits

        if i % 2 == 0:  # Set y-axis labels only for the left two subplots
            ax.set_ylabel('Load (kW)')

        if i >= 2:  # Set x-axis labels and major tick formatting only for the bottom two subplots
            ax.set_xlabel('Time of Day')
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        else:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter(''))
            ax.tick_params(axis='x', which='both', length=0)  # Hide the ticks

        # Adjust the x-axis limit to end where the data ends, removing the last unwanted tick
        last_time = times[-1] if times else arbitrary_date
        ax.set_xlim([arbitrary_date, last_time])

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.xaxis.set_minor_locator(mdates.HourLocator())

        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('./plots/4XLoadProfiles.pdf')
    plt.savefig('/home/laurenz/Documents/DAI/_Thesis/git/Thesis_template/Figures/plots/4XLoadProfiles.pdf')
    plt.show()


# Main function to load data and plot
def main():
    filepath = '../data/load/load_15min_intervals_74_kW.csv'
    df = load_data(filepath)

    # Define an array of profiles and an array of 8 dates for each profile
    profiles = ['profile_2', 'profile_4', 'profile_20', 'profile_68']
    dates_per_profile = [
        ['2010-06-23', '2010-12-23'],  # Dates for profile_17
        ['2010-06-23', '2010-12-23'],  # Dates for profile_18
        ['2010-06-23', '2010-12-23'],  # Dates for profile_19
        ['2010-06-23', '2010-12-23']  # Dates for profile_20
    ]

    # Define colors for summer and winter plots
    colors = ['#9881FB', '#59CC40']  # Adjust the colors as per your preference

    plot_load_profiles(df, dates_per_profile, profiles, colors)


if __name__ == '__main__':
    main()
