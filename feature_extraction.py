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

#def extract_actigraphy_features(actigraphy_data):
#   features_per_minute = {}
#
#   for index, row in actigraphy_data.iterrows():
#       day = row['day']
#       time = row['time']
#
#       # Estraiamo l'ora, i minuti e i secondi dal timestamp
#       timestamp = datetime.strptime(time, '%H:%M:%S')
#       hour = timestamp.hour
#       minute = timestamp.minute
#
#       # Creiamo una chiave univoca per ogni minuto (day, hour, minute)
#       key = (day, hour, minute)
#
#       # Se questa chiave non esiste già, creiamo un dizionario per questa chiave
#       if key not in features_per_minute:
#           features_per_minute[key] = {
#               'Axis1': [],
#               'Axis2': [],
#               'Axis3': [],
#               'Steps': [],
#               'HR': [],
#               'Inclinometer Off': [],
#               'Inclinometer Standing': [],
#               'Inclinometer Sitting': [],
#               'Inclinometer Lying': [],
#               'Vector Magnitude': [],
#           }
#
#       features_per_minute[key]['Axis1'].append(row['Axis1'])
#       features_per_minute[key]['Axis2'].append(row['Axis2'])
#       features_per_minute[key]['Axis3'].append(row['Axis3'])
#       features_per_minute[key]['Steps'].append(row['Steps'])
#       features_per_minute[key]['HR'].append(row['HR'])
#       features_per_minute[key]['Inclinometer Off'].append(row['Inclinometer Off'])
#       features_per_minute[key]['Inclinometer Standing'].append(row['Inclinometer Standing'])
#       features_per_minute[key]['Inclinometer Sitting'].append(row['Inclinometer Sitting'])
#       features_per_minute[key]['Inclinometer Lying'].append(row['Inclinometer Lying'])
#       features_per_minute[key]['Vector Magnitude'].append(row['Vector Magnitude'])
#
#   for key, value in features_per_minute.items():
#       for column in ['Axis1', 'Axis2', 'Axis3', 'Steps', 'HR', 'Inclinometer Off', 'Inclinometer Standing', 'Inclinometer Sitting', 'Inclinometer Lying', 'Vector Magnitude']:
#           data = value[column]
#           if data:
#               value[f'mean_{column.lower()}'] = np.mean(data)
#               value[f'median_{column.lower()}'] = np.median(data)
#               value[f'var_{column.lower()}'] = np.var(data)
#               value[f'std_{column.lower()}'] = np.std(data)
#               value[f'max_{column.lower()}'] = np.max(data)
#               value[f'min_{column.lower()}'] = np.min(data)
#               value[f'entropy_{column.lower()}'] = entropy(data)
#               value[f'skew_{column.lower()}'] = skew(data)
#               value[f'kurtosis_{column.lower()}'] = kurtosis(data)
#               value[f'iqr_{column.lower()}'] = np.percentile(data, 75) - np.percentile(data, 25)
#               value[f'mad_{column.lower()}'] = np.mean(np.abs(np.array(data) - np.mean(data)))
#       
#   return features_per_minute

#def extract_rr_features(rr_data):
#    features_per_minute = {}
#
#    for index, row in rr_data.iterrows():
#        day = row['day']
#        time = row['time']
#
#        # Estraiamo l'ora, i minuti e i secondi dal timestamp
#        timestamp = datetime.strptime(time, '%H:%M:%S')
#        hour = timestamp.hour
#        minute = timestamp.minute
#
#        # Creiamo una chiave univoca per ogni minuto (day, hour, minute)
#        key = (day, hour, minute)
#
#        # Se questa chiave non esiste già, creiamo un dizionario per questa chiave
#        if key not in features_per_minute:
#            features_per_minute[key] = {
#                'ibi_s': [],  # Inizializziamo una lista per raccogliere i valori della colonna 'ibi_s'
#            }
#
#        # Aggiungiamo il valore della colonna 'ibi_s' alla lista corrispondente a questa chiave
#        features_per_minute[key]['ibi_s'].append(row['ibi_s'])
#
#    # Calcoliamo le statistiche per 'ibi_s' per ogni minuto
#    for key, value in features_per_minute.items():
#        data = value['ibi_s']
#        if data:
#            value['mean_ibi_s'] = np.mean(data)
#            value['median_ibi_s'] = np.median(data)
#            value['var_ibi_s'] = np.var(data)
#            value['std_ibi_s'] = np.std(data)
#            value['max_ibi_s'] = np.max(data)
#            value['min_ibi_s'] = np.min(data)
#            value['entropy_ibi_s'] = entropy(data)
#            value['skew_ibi_s'] = skew(data)
#            value['kurtosis_ibi_s'] = kurtosis(data)
#            value['iqr_ibi_s'] = np.percentile(data, 75) - np.percentile(data, 25)
#            value['mad_ibi_s'] = np.mean(np.abs(np.array(data) - np.mean(data)))
#
#            del value['ibi_s']
#
#    print(features_per_minute[1,10,10])
#    return features_per_minute