import os
import pandas as pd
from feature_extraction import extract_actigraphy_features, extract_rr_features, filter_data_by_sleep_intervals
from classifiers import run_classifiers
import csv

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
            #filtered_actigraphy_data, filtered_rr_data = filter_data_by_sleep_intervals(user_folder, current_directory)
            filtered_actigraphy_data, filtered_rr_data = filter_data_by_sleep_intervals('user_2', current_directory)

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
                        'y': y
                    }
                    all_combined_features.append(minute_dict)
        
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(all_combined_features)

    # Write the DataFrame to a CSV file
    df.to_csv('combined_features.csv', index=False)
else:
    print("The combined_features.csv file already exists. No need to run the code.")
    df = pd.read_csv('combined_features.csv')
    
#num_users = df['user_id'].nunique()
users_df = df.groupby(df.user_id)

# Run classifiers on the combined data
#results = run_classifiers(df.drop(columns=['y']), df['y'], num_users)
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
        csv_writer.writerow([model, user_id, classe] + list(values))
