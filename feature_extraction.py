from datetime import datetime
import numpy as np
from scipy.stats import entropy, skew, kurtosis

def extract_actigraphy_features[key](actigraphy_data):
    minute_features = {}

    for index, row in actigraphy_data.iterrows():
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
        for column in ['Axis1', 'Axis2', 'Axis3', 'Steps', 'HR', 'Inclinometer Off', 'Inclinometer Standing', 'Inclinometer Sitting', 'Inclinometer Lying', 'Vector Magnitude']:
            minute_features[key][column].append(row[column])

    # Calculate statistics for each column for each minute
    features_per_minute = {}
    for key, value in minute_features.items():
        features_per_minute[key] = {}
        for column in value:
            data = value[column]
            if data:
                if column in ['Inclinometer Off', 'Inclinometer Standing', 'Inclinometer Sitting', 'Inclinometer Lying']:
                    features_per_minute[key][f'{column.lower()}'] = 1 if any(data) else 0
                else:
                    features_per_minute[key][f'mean_{column.lower()}'] = np.nan_to_num(np.mean(data), nan=0)
                    features_per_minute[key][f'median_{column.lower()}'] = np.nan_to_num(np.median(data), nan=0)
                    features_per_minute[key][f'var_{column.lower()}'] = np.nan_to_num(np.var(data), nan=0)
                    features_per_minute[key][f'std_{column.lower()}'] = np.nan_to_num(np.std(data), nan=0)
                    features_per_minute[key][f'max_{column.lower()}'] = np.nan_to_num(np.max(data), nan=0)
                    features_per_minute[key][f'min_{column.lower()}'] = np.nan_to_num(np.min(data), nan=0)
                    features_per_minute[key][f'entropy_{column.lower()}'] = np.nan_to_num(entropy(data), nan=0)
                    features_per_minute[key][f'skew_{column.lower()}'] = np.nan_to_num(skew(data), nan=0)
                    features_per_minute[key][f'kurtosis_{column.lower()}'] = np.nan_to_num(kurtosis(data), nan=0)
                    features_per_minute[key][f'iqr_{column.lower()}'] = np.nan_to_num(np.percentile(data, 75) - np.percentile(data, 25), nan=0)
                    features_per_minute[key][f'mad_{column.lower()}'] = np.nan_to_num(np.mean(np.abs(np.array(data) - np.mean(data))), nan=0)

    return features_per_minute

def extract_rr_features(rr_data):
    minute_features = {}

    for index, row in rr_data.iterrows():
        day = row['day']
        time = row['time']

        # Extract hour and minute from the timestamp
        hour, minute, _ = map(int, time.split(':'))

        # Create a unique key for each minute (day, hour, minute)
        key = (day, hour, minute)

        # If this key doesn't exist already, create a dictionary for it
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
                'max_ibi_s': np.nan_to_num(np.max(data), nan=0),
                'min_ibi_s': np.nan_to_num(np.min(data), nan=0),
                'entropy_ibi_s': np.nan_to_num(entropy(data), nan=0),
                'skew_ibi_s': np.nan_to_num(skew(data), nan=0),
                'kurtosis_ibi_s': np.nan_to_num(kurtosis(data), nan=0),
                'iqr_ibi_s': np.nan_to_num(np.percentile(data, 75) - np.percentile(data, 25), nan=0),
                'mad_ibi_s': np.nan_to_num(np.mean(np.abs(np.array(data) - np.mean(data))), nan=0)
            }
            
    return features_per_minute