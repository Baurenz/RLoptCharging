import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from plots_thesis import plot_style

# Load the dataset
file_path = '../data/market/Germany_day_ahead.csv'  # Make sure this path is correct for your environment
data = pd.read_csv(file_path)
plot_style.set_plot_style()
plt.rcParams['figure.figsize'] = (10, 4)  # Adjust based on your document's layout and column width

# Convert 'Datetime (UTC)' to datetime format
data['Datetime (UTC)'] = pd.to_datetime(data['Datetime (UTC)'])

# Extract the 'Price (EUR/MWhe)' column and convert to €/kWh
prices = data['Price (EUR/MWhe)'] / 1000

# # Adjust prices according to the formula: (EPEX Spot DE + 3% of EPEX Spot DE) + 0.1671 €
# prices = prices * 1.03 + 0.1671
#
# # Apply the bounds of 80 Cent/kWh and -80 Cent/kWh
# prices = prices.clip(lower=-0.80, upper=0.80)

# Plot a histogram to visually inspect the distribution, showing the fraction of the total
plt.hist(prices, bins=100, color=plot_style.color_cycle[1], edgecolor=plot_style.color_cycle[0], density=True)  # Adjusted bins and range, with density=True
plt.xlabel('Price (EUR/kWh)')
plt.ylabel('Fraction of Total')
plt.grid(True)
plt.xlim(-0.2, 0.4)
plt.savefig(f'./plots/distribution_DA.pdf')
plt.savefig(f'/home/laurenz/Documents/DAI/_Thesis/git/Thesis_template/Figures/plots/distribution_DA.pdf')
plt.show()

# Calculate and print basic statistics for the modified prices
mean_price = prices.mean()
median_price = prices.median()
std_dev_price = prices.std()
skewness = skew(prices)
kurtosis_value = kurtosis(prices, fisher=False)  # Using Pearson's definition for kurtosis

print(f'Mean Price: {mean_price:.4f} EUR/kWh')
print(f'Median Price: {median_price:.4f} EUR/kWh')
print(f'Standard Deviation of Price: {std_dev_price:.4f} EUR/kWh')
print(f'Skewness: {skewness:.2f}')
print(f'Kurtosis: {kurtosis_value:.2f}')

