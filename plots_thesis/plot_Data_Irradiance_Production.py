################################################
# Plot to show Irradiance, Pv Production (with and without noise) for Two days (winter summer) for Data-Part: PV
###############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import plot_style  # Import the custom plot style module

plt.rcParams['figure.figsize'] = (10, 4)  # Adjust based on your document's layout and column width


def load_data(filepath):
    df = pd.read_csv(filepath)
    df['period_end'] = pd.to_datetime(df['period_end']).dt.tz_localize(None)
    return df


def calculate_pv_production(gti, temp_ambient, cloud_opacity, pv_kWp):
    G_standard = 1000
    T_standard = 25
    alpha_T = -0.004

    T_cell = temp_ambient + (T_standard / 800) * gti
    P_PV = pv_kWp * (gti / G_standard) * (1 + alpha_T * (T_cell - T_standard))
    P_PV = max(P_PV, 0)

    # Adjust noise level based on cloud opacity
    if 30 <= cloud_opacity <= 70:
        noise_level = -abs(np.random.normal(loc=0, scale=0.1))  # Ensure noise is negative or zero
    else:
        noise_level = -abs(np.random.normal(loc=0, scale=0.05))  # Ensure noise is negative or zero

    # Apply noise based on cloud opacity
    noise_factor = 1 + noise_level  # This will now decrease the P_PV due to negative noise_level
    P_PV_noise = P_PV * noise_factor  # P_PV_noise will be less than or equal to P_PV
    P_PV_noise = max(P_PV_noise, 0)  # Ensure noise doesn't result in negative production

    return P_PV, P_PV_noise


def plot_data_for_dates(df, dates, pv_kWp):
    plot_style.set_plot_style()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Define colors for the plots
    colors = {'GTI': ['#9881FB', '#59CC40'], 'PV': ['#2090CC', '#444344'], 'P_PV_noise': ['#289792', '#044859']}

    # Lists to collect legend handles and labels
    summer_handles, summer_labels = [], []
    winter_handles, winter_labels = [], []

    for i, date in enumerate(dates):
        start_period = pd.to_datetime(date)
        end_period = start_period + pd.Timedelta(days=1, seconds=-1)

        if i == 0:
            day_label = 'Summer Day'
        elif i == 1:
            day_label = 'Winter Day'
        else:
            day_label = f'Day {i + 1}'

        selected_date_data = df[(df['period_end'] >= start_period) & (df['period_end'] <= end_period)]
        P_PV, P_PV_noise = zip(*[calculate_pv_production(gti, air_temp, cloud_opacity, pv_kWp) for gti, air_temp, cloud_opacity in
                                 zip(selected_date_data['gti'], selected_date_data['air_temp'], selected_date_data['cloud_opacity'])])

        # Convert times for plotting
        common_date = pd.Timestamp('2000-01-01')
        times = [(common_date.replace(hour=t.hour, minute=t.minute, second=t.second)) for t in selected_date_data['period_end'].dt.time]

        # Plotting and collecting handles/labels
        gti_line, = ax1.plot(times, selected_date_data['gti'], color=colors['GTI'][i], linestyle='--', label=f'GTI {day_label}')
        pv_line, = ax2.plot(times, P_PV, color=colors['PV'][i], label=r'$P_{PV}$' + ' ' + day_label)
        pv_noise_line, = ax2.plot(times, P_PV_noise, color=colors['P_PV_noise'][i], linestyle=':',
                                  label=r'$P_{PV,\mathcal{N}}$' + ' ' + day_label)

        # Append handles and labels for the current day to the respective lists
        if 'Summer' in day_label:
            summer_handles.extend([gti_line, pv_line, pv_noise_line])
            summer_labels.extend([gti_line.get_label(), pv_line.get_label(), pv_noise_line.get_label()])
        elif 'Winter' in day_label:
            winter_handles.extend([gti_line, pv_line, pv_noise_line])
            winter_labels.extend([gti_line.get_label(), pv_line.get_label(), pv_noise_line.get_label()])

    summer_handles = summer_handles
    summer_labels = summer_labels

    # Combine the handles and labels for winter
    winter_handles = winter_handles
    winter_labels = winter_labels

    # Add the summer legend to the plot
    summer_legend = ax1.legend(summer_handles, summer_labels, loc='upper right', bbox_to_anchor=(1, 1))
    ax1.add_artist(summer_legend)  # This is necessary to keep the summer legend when adding the next one

    # Create a legend for winter and place it in the middle on the right side of the figure
    winter_legend = ax1.legend(winter_handles, winter_labels, loc='center right', bbox_to_anchor=(1, 0.5))

    ax1.set_xlabel('Time of Day')
    ax1.set_ylabel('GTI Irradiance | W/m2', color='black')
    ax2.set_ylabel('PV Production | kW', color='black')

    # Formatting for the x-axis (time)
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(1, 24, 2)))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Explicitly set the alignment of the x-tick labels
    for label in ax1.get_xticklabels():
        label.set_ha('center')  # 'ha' is short for horizontalalignment
        label.set_rotation(0)  # Set rotation to 0 degrees (horizontal)

    # Limit x-axis from 00:00 to 00:00 of the next day
    start_time = pd.Timestamp('2000-01-01 00:00:00')
    end_time = pd.Timestamp('2000-01-02 00:00:00')

    def custom_formatter(x, pos):
        # Convert to a datetime object
        dt = mdates.num2date(x).replace(tzinfo=None)
        # Check if it's the last tick, comparing to 'end_time'
        if dt.strftime('%Y-%m-%d %H:%M') == end_time.strftime('%Y-%m-%d %H:%M'):
            return '24:00'
        else:
            return dt.strftime('%H:%M')

    # Apply the custom formatter
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))

    ax1.set_xlim(start_time, end_time)

    # Remove the plot title
    # ax1.set_title('')  # No need to set an empty title, just don't include plt.title() at all

    ax1.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig('./plots/Data_Irradiance_Production2.pdf')
    plt.savefig('/home/laurenz/Documents/DAI/_Thesis/git/Thesis_template/Figures/plots/Data_Irradiance_Production2.pdf')
    plt.show()


def main():
    filepath = '../create_data/solar_irradiance/weather_data_reilingen.csv'
    df = load_data(filepath)
    dates = ['2022-06-18', '2022-12-18']  # Dates for which to plot data
    pv_kWp = 15
    plot_data_for_dates(df, dates, pv_kWp)


if __name__ == '__main__':
    main()
