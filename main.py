# Description: This script generates a scatter plot of the number of data
# packages published to the EDI repository, and estimates the curation capacity
# required to keep up with the rate of data submissions.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def plot_data(data_file, plot_title, output_file, high_quality_max = None, low_quality_max = None, show_capacity=False, ):
    """Generate a scatter plot of the number of data packages published to the
    EDI repository, and estimate the curation capacity required to keep up with
    the rate of data submissions.
    :param data_file: The path to a CSV file containing data on the number of
        data packages published to the EDI repository by the data curation
        team. Has columns 'pid' (e.g. value 'edi.1.1'), `datetime` (e.g. value
        '2016-12-01 12:55:09'), principal (e.g. value
        'uid=EDI,o=EDI,dc=edirepository,dc=org').
    :param high_quality_max: The maximum number of packages that can be curated
        per week, when applying high quality effort.
    :param low_quality_max: The maximum number of packages that can be curated
        per week, when applying bare minimum effort.
    :param show_capacity: A boolean indicating whether to show the curation
    :param plot_title: The title of the plot
    :param output_file: The path to the output file where the plot will be
        saved.
    """
    # Parameterize this function
    rolling_window = 13  # units = weeks
    outlier_threshold = 17  # remove values over this threshold
    high_quality_high = high_quality_max  # the maximum number of packages that can be curated per week, when applying high quality effort
    low_quality_high = low_quality_max  # maximum number of packages that can be curated per week, when applying bare minimum effort

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
    plt.ylabel("Number of Packages / Week")
    if show_capacity:
        plt.title(plot_title, loc='left')
        plt.yticks(np.arange(0, weekly_data.max() + 18, 2))
    else:
        plt.title(plot_title, loc='left')
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
    plt.savefig(output_file, dpi=300)

    # display the plot
    plt.show()


def average_curation_effort(data_file, high_quality_multiplier=1, low_quality_multiplier=0.5, include_std=True):
    """Calculate the average curation effort for new data packages and updates
        to data packages
    :param data_file: The path to the data curation log file in .tsv format
    :param high_quality_multiplier: A multiplier for the high quality curation
        effort.
    :param low_quality_multiplier: A multiplier for the low quality curation
        effort.
    :param include_std: A boolean indicating whether to include the standard
        deviation in the calculation of the average curation effort.
    :return: A dictionary containing the average curation effort for new data
        packages and updates to data packages. The keys are 'new_high_quality',
        'new_low_quality', 'update_high_quality', and 'update_low_quality
    """
    data = pd.read_csv(data_file, sep='\t')
    # Calculate the average curation effort for new data packages and updates
    # to data packages
    conditions = ['update', 'new']
    res = {}
    for condition in conditions:
        # We can identify updates by looking for the string "update" in the
        # "notes" column. We have to do the opposite for new data packages.
        update_rows = data['notes'].str.contains(condition, case=False, na=False)
        if condition == 'update':
            rows = update_rows
        else:
            rows = ~update_rows
        # Ignore rows of NA values, and unconventionally curated data
        rows = rows & ~data['curation_effort'].isna()
        rows = rows & (data['curation_effort'] <= 5)
        effort = data.loc[rows, 'curation_effort'].mean()
        effort_std = data.loc[rows, 'curation_effort'].std()
        # Calculate the effort for high quality and low quality curation
        high_quality = effort*high_quality_multiplier
        low_quality = effort*low_quality_multiplier
        # Add the standard deviation to the effort if requested
        if include_std:
            high_quality += effort_std
            low_quality += effort_std
        print(f'Average curation effort for {condition} (high quality): {high_quality:.2f}')
        print(f'Average curation effort for {condition} (low quality): {low_quality:.2f}')
        res[f'{condition}_high_quality'] = high_quality
        res[f'{condition}_low_quality'] = low_quality
    return res


def average_capacity(core_max_hrs, core_min_hrs, ancillary_max_hrs, ancillary_min_hrs, effort):
    """Calculate the average curation capacity for a given range of hours and
    the average curation effort for new data packages and updates to data
    packages
    :param core_max_hrs: The maximum number of hours that the core team members
        can work in a week.
    :param core_min_hrs: The minimum number of hours that the core team members
        can work in a week.
    :param ancillary_max_hrs: The maximum number of hours that the ancillary
        team members can work in a week.
    :param ancillary_min_hrs: The minimum number of hours that the ancillary
        team members can work in a week.
    :param effort: A dictionary containing the average curation effort for new
        data packages and updates to data packages. This is the return value of
        the `average_curation_effort` function. The keys are expected to be
        'new_high_quality', 'new_low_quality', 'update_high_quality', and
        'update_low_quality'.
    :return: A pandas DataFrame containing the average curation capacity for
        the core and ancillary teams, noting the high and low quality curation
        efforts. The columns are 'high_quality_capacity' and
        'low_quality_capacity'. The rows are 'core_new', 'core_update',
        'ancillary_new', 'ancillary_update', 'total_new', and 'total_update'.
    """
    # Calculate the average curation capacity for the core team
    core_max_new_high = core_max_hrs / effort['new_high_quality']
    core_max_new_low = core_max_hrs / effort['new_low_quality']
    core_min_new_high = core_min_hrs / effort['new_high_quality']
    core_min_new_low = core_min_hrs / effort['new_low_quality']
    core_max_update_high = core_max_hrs / effort['update_high_quality']
    core_max_update_low = core_max_hrs / effort['update_low_quality']
    core_min_update_high = core_min_hrs / effort['update_high_quality']
    core_min_update_low = core_min_hrs / effort['update_low_quality']

    # Calculate the average curation capacity for the ancillary team
    ancillary_max_new_high = ancillary_max_hrs / effort['new_high_quality']
    ancillary_max_new_low = ancillary_max_hrs / effort['new_low_quality']
    ancillary_min_new_high = ancillary_min_hrs / effort['new_high_quality']
    ancillary_min_new_low = ancillary_min_hrs / effort['new_low_quality']
    ancillary_max_update_high = ancillary_max_hrs / effort['update_high_quality']
    ancillary_max_update_low = ancillary_max_hrs / effort['update_low_quality']
    ancillary_min_update_high = ancillary_min_hrs / effort['update_high_quality']
    ancillary_min_update_low = ancillary_min_hrs / effort['update_low_quality']

    # Calculate the average curation capacity for the entire team
    total_max_new_high = core_max_new_high + ancillary_max_new_high
    total_max_new_low = core_max_new_low + ancillary_max_new_low
    total_min_new_high = core_min_new_high + ancillary_min_new_high
    total_min_new_low = core_min_new_low + ancillary_min_new_low
    total_max_update_high = core_max_update_high + ancillary_max_update_high
    total_max_update_low = core_max_update_low + ancillary_max_update_low
    total_min_update_high = core_min_update_high + ancillary_min_update_high
    total_min_update_low = core_min_update_low + ancillary_min_update_low

    res = pd.DataFrame({
        'high_quality_capacity': [
            core_max_new_high,
            core_min_new_high,
            core_max_update_high,
            core_min_update_high,
            ancillary_max_new_high,
            ancillary_min_new_high,
            ancillary_max_update_high,
            ancillary_min_update_high,
            total_max_new_high,
            total_min_new_high,
            total_max_update_high,
            total_min_update_high,

        ],
        'low_quality_capacity': [
            core_max_new_low,
            core_min_new_low,
            core_max_update_low,
            core_min_update_low,
            ancillary_max_new_low,
            ancillary_min_new_low,
            ancillary_max_update_low,
            ancillary_min_update_low,
            total_max_new_low,
            total_min_new_low,
            total_max_update_low,
            total_min_update_low,
        ]
    }, index=[
        'core_max_new',
        'core_min_new',
        'core_max_update',
        'core_min_update',
        'anciallary_max_new',
        'anciallary_min_new',
        'anciallary_max_update',
        'anciallary_min_update',
        'total_max_new',
        'total_min_new',
        'total_max_update',
        'total_min_update',
        ])
    return res




if __name__ == "__main__":

    effort = average_curation_effort(
        data_file="data_submission_log.tsv",
        include_std=True
    )

    capacity = average_capacity(
        core_max_hrs=10,
        core_min_hrs=9,
        ancillary_max_hrs=15.0,
        ancillary_min_hrs=6.4,
        effort=effort
    )

    # Plot Data Submissions Over Time
    plot_data(
        data_file="curator_published_data_packages.csv",
        high_quality_max=capacity.loc['core_max_new', 'high_quality_capacity'],
        low_quality_max=capacity.loc['core_max_new', 'low_quality_capacity'],
        show_capacity=False,
        plot_title="Data Submissions Over Time",
        output_file='data_submissions_over_time.png'
    )

    # Plot Data Submissions Over Time and Core Curation Team Capacities
    plot_data(
        data_file="curator_published_data_packages.csv",
        high_quality_max=capacity.loc['core_max_new', 'high_quality_capacity'],
        low_quality_max=capacity.loc['core_max_new', 'low_quality_capacity'],
        show_capacity=True,
        plot_title="Data Submissions Over Time and Core Curation Team Capacities",
        output_file="data_submissions_with_core_capacity.png"
    )

    # Plot Data Submissions Over Time and Core + Ancillary Curation Team Capacities
    plot_data(
        data_file="curator_published_data_packages.csv",
        high_quality_max=capacity.loc['total_max_new', 'high_quality_capacity'],
        low_quality_max=capacity.loc['total_max_new', 'low_quality_capacity'],
        show_capacity=True,
        plot_title="Data Submissions Over Time and Core + Ancillary Curation Team Capacities",
        output_file="data_submissions_with_core_and_ancillary_capacity.png"
    )


