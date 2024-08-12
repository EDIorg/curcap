# Description: This script generates a scatter plot of the number of data
# packages published to the EDI repository, and estimates the curation capacity
# required to keep up with the rate of data submissions.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def plot_data(data_file, show_capacity=False):
    # Parameterize this function
    rolling_window = 13  # units = weeks
    outlier_threshold = 17  # remove values over this threshold
    high_quality_high = 8.74  # the maximum number of packages that can be curated per week, when applying high quality effort
    low_quality_high = 18  # maximum number of packages that can be curated per week, when applying bare minimum effort

    # Sample data (replace with your actual CSV data)
    data = pd.read_csv(data_file)

    # Convert datetime a DatetimeIndex
    data['datetime'] = pd.to_datetime(data['datetime'])

    # Group data by week and calculate total applications
    weekly_data = data.groupby(pd.Grouper(key='datetime', freq='W')).size()

    # Remove outliers; values over 17. These correspond with high rates of
    # ecocomDP package publications, which are not representative of the
    # typical data curation workload.
    weekly_data = weekly_data[weekly_data < outlier_threshold]

    # Create the scatter plot
    plt.figure(figsize=(12, 6))
    plt.scatter(weekly_data.index, weekly_data.values, color='grey', s=2.5, alpha=0.5)

    # Format x-axis as dates
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add moving average
    weekly_data.rolling(window=rolling_window, center=True).mean().plot(color='grey', linewidth=2)

    # Fit trend line to data
    x = np.arange(len(weekly_data))
    y = weekly_data.values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(weekly_data.index, p(x), "r--")

    # # Remove zero values from the data for exponential trend line
    # weekly_data = weekly_data[weekly_data > 0]
    #
    # # Fit exponential trend line to data
    # x = np.arange(len(weekly_data))
    # y = weekly_data.values
    # z = np.polyfit(x, np.log(y), 1)
    # p = np.poly1d(z)
    # plt.plot(weekly_data.index, np.exp(p(x)), "b--")
    #
    # # Determine if the linear or exponential fit is better (very rough estimate)
    # linear_fit = np.polyfit(x, y, 1)
    # exponential_fit = np.polyfit(x, np.log(y), 1)
    # linear_residuals = y - np.polyval(linear_fit, x)
    # exponential_residuals = y - np.exp(np.polyval(exponential_fit, x))
    # linear_sse = np.sum(linear_residuals ** 2)
    # exponential_sse = np.sum(exponential_residuals ** 2)
    # print(f'Linear SSE: {linear_sse}')
    # print(f'Exponential SSE: {exponential_sse}')



    # Add labels, tick marks, and title
    plt.xlabel('Week')
    plt.ylabel("Number of Packages")
    if show_capacity:
        plt.title("Data Submissions Over Time and Current Curation Capacities", loc='left')
        plt.yticks(np.arange(0, weekly_data.max() + 6, 2))
    else:
        plt.title("Data Submissions Over Time", loc='left')
        plt.yticks(np.arange(0, weekly_data.max() + 1, 2))

    if show_capacity:
        # Create a DateTimeIndex for the shaded regions beggining 2023-06-01 and ending at the last date in the data
        date_range = pd.date_range(start='2023-06-01', end=weekly_data.index[-1], freq='W')

        # Create stack of shaded regions to represent different curation capacities, beginning 2023-06-01
        plt.fill_between(date_range, 0, high_quality_high, color='green', alpha=0.1)
        plt.fill_between(date_range, high_quality_high, low_quality_high, color='red', alpha=0.1)

        # Add key for curation capacities
        plt.legend(['_Packages/Week', '13-Week Moving Average', 'Trend Line' + f' (y = {z[0]:.2f}x + {z[1]:.2f})', 'High Quality Curation', 'Low Quality Curation'])
    else:
        # Add key
        plt.legend(['_Packages/Week', '13-Week Moving Average',
                    'Trend Line' + f' (y = {z[0]:.2f}x + {z[1]:.2f})'])

    # Misc metrics: Calculate number of weeks passed for a date since the start of the data
    start_date = weekly_data.index[0]
    end_date = '2028-10-01'
    weeks_passed = (pd.to_datetime(end_date) - start_date).days / 7
    print(f'Weeks passed since the start of the data: {weeks_passed})')

    # Misc metrics: Calculate y value of trend line for a given x value
    y = 0.01 * weeks_passed + 0.85
    print(f'Number of packages expected for {end_date}: {y:.1f}')

    # Write the plot to a file
    if show_capacity:
        plt.savefig('/Users/csmith/Desktop/curation_capacity_analysis_with_capacity.png', dpi=300)
    else:
        plt.savefig('/Users/csmith/Desktop/curation_capacity_analysis.png',
                    dpi=300)

    # display the plot
    plt.show()


# Test the functions from main
if __name__ == "__main__":
    # get_eml("/Users/csmith/Data/edi/all_edi_scope_eml")
    # data = get_data("/Users/csmith/Data/edi/all_eml")  # len = 749
    plot_data("/Users/csmith/Data/edi/curator_published_data_packages.csv", show_capacity=True)

