import numpy as np
import pandas as pd
from scipy.stats import entropy, skew, kurtosis
import os


def extract_actigraphy_features(actigraphy_data):
    """
    Extracts detailed features from actigraphy data for each minute, including activity counts on three axes,
    step count, heart rate, inclinometer positions, and vector magnitude. S
    Statistical summaries are computed for each minute across all the specified features.

    Parameters
    ----------
    actigraphy_data : DataFrame
        A pandas DataFrame containing the actigraphy data with columns for day, time, and various sensor
        readings like Axis1, Axis2, Axis3, Steps, HR (Heart Rate), inclinometer statuses, and vector magnitude.

    Returns
    -------
    features_per_minute : dict
        A dictionary keyed by (day, hour, minute) tuples, containing summarized statistics (mean, std, entropy, etc.)
        for each minute across various sensor features.
    """
    # Initialize a dictionary to store features per minute
    minute_features = {}

    # Iterate through each row of actigraphy data
    for index, row in actigraphy_data.iterrows():
        # Extract day and time from the row
        day = row['day']
        time = row['time']

        # Extract hour and minute from the timestamp
        hour, minute, _ = map(int, time.split(':'))

        # Create a unique key for each minute (day, hour, minute)
        key = (day, hour, minute)

        # If this key doesn't exist already, create a dictionary for it
        if key not in minute_features:
            minute_features[key] = {
                'Axis1': [],
                'Axis2': [],
                'Axis3': [],
                'Steps': [],
                'HR': [],
                'Inclinometer Off': [],
                'Inclinometer Standing': [],
                'Inclinometer Sitting': [],
                'Inclinometer Lying': [],
                'Vector Magnitude': [],
            }

        # Append the values to the corresponding minute dictionary
        for column in ['Axis1', 'Axis2', 'Axis3', 'Steps', 'HR', 'Inclinometer Off', 'Inclinometer Standing',
                       'Inclinometer Sitting', 'Inclinometer Lying', 'Vector Magnitude']:
            minute_features[key][column].append(row[column])

    # Calculate statistics for each column for each minute
    features_per_minute = {}
    for key, value in minute_features.items():
        features_per_minute[key] = {}
        for column in value:
            data = value[column]
            if data:
                if column in ['Inclinometer Off', 'Inclinometer Standing', 'Inclinometer Sitting',
                              'Inclinometer Lying']:
                    features_per_minute[key][f'{column.lower()}'] = 1 if any(data) else 0
                else:
                    features_per_minute[key][f'mean_{column.lower()}'] = np.nan_to_num(np.mean(data), nan=0)
                    features_per_minute[key][f'median_{column.lower()}'] = np.nan_to_num(np.median(data), nan=0)
                    features_per_minute[key][f'var_{column.lower()}'] = np.nan_to_num(np.var(data), nan=0)
                    features_per_minute[key][f'std_{column.lower()}'] = np.nan_to_num(np.std(data), nan=0)
                    # features_per_minute[key][f'max_{column.lower()}'] = np.nan_to_num(np.max(data), nan=0)
                    # features_per_minute[key][f'min_{column.lower()}'] = np.nan_to_num(np.min(data), nan=0)
                    features_per_minute[key][f'entropy_{column.lower()}'] = np.nan_to_num(entropy(data), nan=0)
                    features_per_minute[key][f'skew_{column.lower()}'] = np.nan_to_num(skew(data), nan=0)
                    features_per_minute[key][f'kurtosis_{column.lower()}'] = np.nan_to_num(kurtosis(data), nan=0)
                    features_per_minute[key][f'iqr_{column.lower()}'] = np.nan_to_num(
                        np.percentile(data, 75) - np.percentile(data, 25), nan=0)
                    features_per_minute[key][f'mad_{column.lower()}'] = np.nan_to_num(
                        np.mean(np.abs(np.array(data) - np.mean(data))), nan=0)

    return features_per_minute


def extract_rr_features(rr_data):
    """
    Extracts and computes statistical features from RR data for each minute.
    The features include mean, median, variance, standard deviation, entropy, skewness, kurtosis, interquartile range,
    and mean absolute deviation of the interbeat intervals.

    Parameters
    ----------
    rr_data : DataFrame
        A pandas DataFrame containing RR data with columns for day, time, and interbeat intervals ('ibi_s').

    Returns
    -------
    features_per_minute : dict
        A dictionary keyed by (day, hour, minute) tuples, containing the calculated statistical features for
        the 'ibi_s' values for each minute. The dictionary includes entries for mean, median, variance,
        standard deviation, entropy, skewness, kurtosis, interquartile range, and mean absolute deviation.

    Notes
    -----
    This function iterates through each row of the RR data, grouping the data by minute based on the 'day', 'hour',
    and 'minute' extracted from the timestamp. It calculates several statistics for the interbeat intervals for each minute.
    """
    # Initialize a dictionary to store features per minute
    minute_features = {}

    # Iterate through each row of RR data
    for index, row in rr_data.iterrows():
        # Extract day and time from the row
        day = row['day']
        time = row['time']

        # Extract hour and minute from the timestamp
        hour, minute, _ = time.split(':')

        # Create a unique key for each minute (day, hour, minute)
        key = tuple(map(int, (day, hour, minute)))

        # If this key doesn't exist already, create a list for it
        if key not in minute_features:
            minute_features[key] = []

        # Append the 'ibi_s' value to the corresponding minute list
        minute_features[key].append(row['ibi_s'])

    # Calculate statistics for 'ibi_s' for each minute
    features_per_minute = {}
    for key, data in minute_features.items():
        if data:
            features_per_minute[key] = {
                'mean_ibi_s': np.nan_to_num(np.mean(data), nan=0),
                'median_ibi_s': np.nan_to_num(np.median(data), nan=0),
                'var_ibi_s': np.nan_to_num(np.var(data), nan=0),
                'std_ibi_s': np.nan_to_num(np.std(data), nan=0),
                # 'max_ibi_s': np.nan_to_num(np.max(data), nan=0),
                # 'min_ibi_s': np.nan_to_num(np.min(data), nan=0),
                'entropy_ibi_s': np.nan_to_num(entropy(data), nan=0),
                'skew_ibi_s': np.nan_to_num(skew(data), nan=0),
                'kurtosis_ibi_s': np.nan_to_num(kurtosis(data), nan=0),
                'iqr_ibi_s': np.nan_to_num(np.percentile(data, 75) - np.percentile(data, 25), nan=0),
                'mad_ibi_s': np.nan_to_num(np.mean(np.abs(np.array(data) - np.mean(data))), nan=0)
            }

    return features_per_minute


def filter_data_by_sleep_intervals(user_folder, current_directory):
    """
    Filters actigraphy and RR data based on sleep intervals defined in a user-specific sleep data file.
    This method ensures that only data falling within these defined sleep periods are considered in subsequent analysis.

    Parameters
    ----------
    user_folder : str
        The folder name for the user whose data is being analyzed. This folder is expected to contain the relevant CSV files.
    current_directory : str
        The path to the directory where the user's folder is located.

    Returns
    -------
    filtered_actigraphy_data : DataFrame
        Actigraphy data filtered to include only entries that fall within the sleep periods specified in the user's sleep data.
    filtered_rr_data : DataFrame
        RR data filtered in the same way, including only the entries within the defined sleep periods.

    Notes
    -----
    The function reads 'sleep.csv', 'Actigraph.csv', and 'RR.csv' from the specified user folder.
    It processes each sleep interval to extract relevant portions of actigraphy and RR data,
    ensuring the data corresponds strictly to periods where the user was reported to be in bed.
    Data from periods that span midnight are handled to ensure continuity and accuracy.
    """
    # Load sleep data from the sleep.csv file to obtain sleep intervals for this user
    sleep_data = pd.read_csv(os.path.join(current_directory, user_folder, "sleep.csv"))

    # Load data from Actigraph.csv and RR.csv files
    actigraphy_data = pd.read_csv(os.path.join(current_directory, user_folder, "Actigraph.csv"))
    rr_data = pd.read_csv(os.path.join(current_directory, user_folder, "RR.csv"))

    # Initialize lists to store filtered data
    filtered_actigraphy_data = []
    filtered_rr_data = []

    # Iterate through rows of sleep_data to obtain sleep intervals
    for _, row in sleep_data.iterrows():
        in_bed_date = row["In Bed Date"]
        in_bed_time = row["In Bed Time"]
        out_bed_date = row["Out Bed Date"]
        out_bed_time = row["Out Bed Time"]

        # Filter actigraphy data based on sleep intervals
        filtered_actigraphy_data.append(
            actigraphy_data[
                (
                        (actigraphy_data[
                             'day'] == in_bed_date) &  # Data is on the same day as the start of the sleep period
                        (actigraphy_data['time'] >= in_bed_time) &  # Data after the start of the sleep period
                        (actigraphy_data['time'] <= out_bed_time)  # Data before the end of the sleep period
                ) |
                (
                        (in_bed_date != out_bed_date) &  # The sleep period runs through midnight
                        (
                                (
                                        (actigraphy_data[
                                             'day'] == in_bed_date) &  # Data after the start of the sleep period
                                        (actigraphy_data['time'] >= in_bed_time)
                                    # Data is on the day of the end of the sleep period
                                ) |
                                (
                                        (actigraphy_data[
                                             'day'] == out_bed_date) &  # Data is on the day of the end of the sleep period
                                        (actigraphy_data['time'] <= out_bed_time)
                                    # Data before the end of the sleep period
                                )
                        )
                )
                ]
        )

        # Filter RR data based on sleep intervals
        filtered_rr_data.append(
            rr_data[
                (
                        (rr_data['day'] == in_bed_date) &  # Data is on the same day as the start of the sleep period
                        (rr_data['time'] >= in_bed_time) &  # Data after the start of the sleep period
                        (rr_data['time'] <= out_bed_time)  # Data before the end of the sleep period
                ) |
                (
                        (in_bed_date != out_bed_date) &  # The sleep period runs through midnight
                        (
                                (
                                        (rr_data['day'] == in_bed_date) &  # Data is on the day the sleep period began
                                        (rr_data['time'] >= in_bed_time)  # Data after the start of the sleep period
                                ) |
                                (
                                        (rr_data[
                                             'day'] == out_bed_date) &  # Data is on the day of the end of the sleep period
                                        (rr_data['time'] <= out_bed_time)  # Data before the end of the sleep period
                                )
                        )
                )
                ]
        )

    # Concatenate the filtered data into a single DataFrame for each sensor
    filtered_actigraphy_data = pd.concat(filtered_actigraphy_data)
    filtered_rr_data = pd.concat(filtered_rr_data)

    return filtered_actigraphy_data, filtered_rr_data


def process_rr_data(rr_data, output_csv):
    all_combined_features = []

    # Extract features for each user, day, hour, and minute
    for user_id, user_data in rr_data.groupby('user_id'):
        y = user_data['y'].iloc[0]  # Assuming 'y' is constant per user

        minute_features = {}
        for index, row in user_data.iterrows():
            day = row['day']
            time = row['time']
            hour, minute, _ = time.split(':')
            key = tuple(map(int, (day, hour, minute)))
            if key not in minute_features:
                minute_features[key] = []
            minute_features[key].append(row['ibi_s'])

        # Calculate statistics for 'ibi_s' for each minute
        for minute_key, data in minute_features.items():
            if data:
                data = np.array(data)
                minute_dict = {
                    #'user_id': user_id,
                    #'day': minute_key[0],
                    #'hour': minute_key[1],
                    #'minute': minute_key[2],
                    'mean_ibi_s': np.nan_to_num(np.mean(data), nan=0),
                    'median_ibi_s': np.nan_to_num(np.median(data), nan=0),
                    'var_ibi_s': np.nan_to_num(np.var(data), nan=0),
                    'std_ibi_s': np.nan_to_num(np.std(data), nan=0),
                    'entropy_ibi_s': np.nan_to_num(entropy(data), nan=0),
                    'skew_ibi_s': np.nan_to_num(skew(data), nan=0),
                    'kurtosis_ibi_s': np.nan_to_num(kurtosis(data), nan=0),
                    'iqr_ibi_s': np.nan_to_num(np.percentile(data, 75) - np.percentile(data, 25), nan=0),
                    'mad_ibi_s': np.nan_to_num(np.mean(np.abs(data - np.mean(data))), nan=0),
                    'y': int(y)
                }
                all_combined_features.append(minute_dict)

    # Convert list of dictionaries to DataFrame and save to CSV
    df_combined = pd.DataFrame(all_combined_features)
    df_combined.to_csv(output_csv, index=False)