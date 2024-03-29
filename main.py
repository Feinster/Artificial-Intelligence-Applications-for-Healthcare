import os
import pandas as pd
from feature_extraction import extract_actigraphy_features, extract_rr_features, filter_data_by_sleep_intervals
from classifiers import run_classifiers
import csv
from feature_selection import (rfe_feature_selection,
                               random_forest_feature_selection,
                               extra_trees_feature_selection,
                               information_gain_feature_selection,
                               variance_threshold_feature_selection)
from config_loader import ConfigLoader
import pgmpy.estimators as estimators
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from sklearn.preprocessing import StandardScaler

config = ConfigLoader.get_instance()
value = config.get('feature.selection').data
write_class_value = config.get('write.class').data
write_classes = [int(c) for c in write_class_value.split(',')]

# Create an empty DataFrame
df = pd.DataFrame()

# Check if the combined_features.csv file exists in the current directory
if not os.path.exists("combined_features.csv"):
    current_directory = os.getcwd()

    all_combined_features = []  # List to store combined features for all users

    # Loop through folders for each user
    for user_folder in os.listdir(current_directory):
        if user_folder.startswith("user_") and os.path.isdir(user_folder):
            # Extract user ID from folder name
            user_id = int(user_folder.split("_")[1])

            # Load questionnaire data for the user
            questionnaire_data = pd.read_csv(os.path.join(current_directory, user_folder, "questionnaire.csv"))

            # Encode labels based on PSQI score
            y = (questionnaire_data['Pittsburgh'] <= 5).astype(int).to_string(index=False)

            # Filter actigraphy and RR data by sleep intervals
            filtered_actigraphy_data, filtered_rr_data = filter_data_by_sleep_intervals(user_folder, current_directory)

            # Extract features from actigraphy data
            actigraphy_features = extract_actigraphy_features(filtered_actigraphy_data)

            # Extract features from RR interval
            rr_features = extract_rr_features(filtered_rr_data)

            # Direct combination of features for each minute
            for minute_key in actigraphy_features.keys():
                if minute_key in rr_features.keys():  # Make sure there is data for both sensors for this minute
                    minute_dict = {
                        'user_id': user_id,
                        'day': minute_key[0],
                        'hour': minute_key[1],
                        'minute': minute_key[2],
                        **actigraphy_features[minute_key],  # Add all actigraphy features
                        **rr_features[minute_key],  # Add all RR interval features
                        'y': int(y)
                    }
                    all_combined_features.append(minute_dict)

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(all_combined_features)

    # Write the DataFrame to a CSV file
    df.to_csv('combined_features.csv', index=False)
else:
    print("The combined_features.csv file already exists. No need to run the code.")
    df = pd.read_csv('combined_features.csv')

# PROVA
# Feature selection
# X = df.drop(columns=['user_id', 'day', 'hour', 'minute', 'y'])
# y = df['y']
# selected_features = rfe_feature_selection(X, y)
# df_selected = df[['user_id', 'day', 'hour', 'minute', 'y'] + selected_features.tolist()]
# users_df = df_selected.groupby(df_selected.user_id)
# FINE PROVA
scaler = StandardScaler()
scaler.fit(df)
df_scaled = scaler.transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
dag = estimators.HillClimbSearch(data=df_scaled).estimate(scoring_method='k2score', max_indegree=4, max_iter=1000,
                                                          show_progress=True)
network = BayesianNetwork(dag)
network.fit(df_scaled)
model = BayesianModelSampling(network)
model.forward_sample(50)

users_df = df.groupby(df.user_id)

# Run classifiers on the combined data
results = run_classifiers(users_df)

# Assuming 'output.csv' is the name of the CSV file where you want to save the data
csv_file_path = 'output.csv'

with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write header row
    csv_writer.writerow(['Model', 'User_id', 'Class', 'Acc', 'Precision', 'Recall', 'F1-score'])

    # Write data rows
    for key, values in results.items():
        model, user_id, classe = key
        if classe in write_classes:
            csv_writer.writerow([model, user_id, classe] + list(values))
