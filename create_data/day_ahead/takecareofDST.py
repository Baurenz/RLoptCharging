import pandas as pd

def clean_dataset():
    # Load the dataset
    df = pd.read_csv('Germany.csv')

    # Ensure 'Datetime (Local)' is in datetime format and set as the index
    df['Datetime (Local)'] = pd.to_datetime(df['Datetime (Local)'])
    df.set_index('Datetime (Local)', inplace=True)

    # Sort the DataFrame by index to ensure chronological order
    df.sort_index(inplace=True)

    # Handle duplicate hours in autumn (fall back) by averaging the prices
    df = df.groupby(df.index).mean()

    # Create a complete datetime index from the start to the end of the dataset
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')

    # Reindex the DataFrame using the complete datetime index, introducing NaNs for any missing times
    df = df.reindex(full_index)

    # Interpolate missing values for spring (spring forward) where there is a missing hour
    df.interpolate(method='time', inplace=True)

    # Reset the index to move 'Datetime (Local)' back to a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Datetime (Local)'}, inplace=True)

    # Assuming 'Country' and 'ISO3 Code' are consistent and can be re-added as shown
    df['Country'] = 'Germany'
    df['ISO3 Code'] = 'DEU'
    # Assuming 'Datetime (UTC)' can be derived by subtracting 1 hour from 'Datetime (Local)'
    df['Datetime (UTC)'] = df['Datetime (Local)'] - pd.Timedelta(hours=1)

    # Select and reorder the columns as per the original format
    df = df[['Country', 'ISO3 Code', 'Datetime (UTC)', 'Datetime (Local)', 'Price (EUR/MWhe)']]

    # Save the cleaned dataset
    df.to_csv('Germany_new.csv', index=False)
    print('Dataset cleaned and saved as Germany_new.csv')

if __name__ == "__main__":
    clean_dataset()
