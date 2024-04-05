import os
import pandas as pd
from feature_extraction import extract_actigraphy_features, extract_rr_features, filter_data_by_sleep_intervals
from classifiers import run_classifiers, run_classifiers_after_deep, write_results_to_csv
from feature_selection import perform_feature_selection_method
from config_loader import ConfigLoader
from sklearn.preprocessing import StandardScaler
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.lib.utils import display_bayesian_network
from sklearn.model_selection import train_test_split


def main():
    config = ConfigLoader.get_instance()
    write_class_value = config.get('write.class').data
    write_classes = [int(c) for c in write_class_value.split(',')]

    df = check_and_process_combined_features()

    train_data, test_data, x_train, y_train, x_test, y_test, selected_train_features = prepare_data(df, config)

    create_and_save_bayesian_network()
    generate_and_save_synthetic_data()

    # lancio i classificatori, passando i dati sintetici come train set,
    # mentre uso il 30 lasciato all'inizio come test set
    df_after_deep = pd.read_csv('synthetic_data.csv')
    balanced_sample = balance_data(df_after_deep, 'y')

    x_train_after_deep = balanced_sample.drop(columns=['y'])
    y_train_after_deep = balanced_sample['y']

    test_data_user_id = test_data.user_id
    test_data = test_data[['y'] + selected_train_features.tolist()]
    df_test = test_data.groupby(test_data_user_id)

    results_after_deep = run_classifiers_after_deep(x_train_after_deep, y_train_after_deep, df_test)
    write_results_to_csv("output_synthetic.csv", results_after_deep, write_classes)

    # lancio i classificatori passando il 70% iniziale come train set
    # mentre uso il 30 lasciato all'inizio come test set
    results_after_deep = run_classifiers_after_deep(x_train, y_train, df_test)
    write_results_to_csv("output_no_synthetic.csv", results_after_deep, write_classes)

    # lancio i classificatori senza operazioni di deep learning
    users_df = df.groupby(df.user_id)
    # Run classifiers on the combined data
    results = run_classifiers(users_df)
    write_results_to_csv("output_basic_oversampled.csv", results, write_classes)


def check_and_process_combined_features():
    # Check if the combined_features.csv file exists in the current directory
    if not os.path.exists("combined_features.csv"):
        print("Processing features...")
        return process_user_folders()
    else:
        print("The combined_features.csv file already exists. No need to run the code.")
        return pd.read_csv('combined_features.csv')


def process_user_folders():
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
            filtered_actigraphy_data, filtered_rr_data = filter_data_by_sleep_intervals(user_folder,
                                                                                        current_directory)

            # Extract features from actigraphy data
            actigraphy_features = extract_actigraphy_features(filtered_actigraphy_data)

            # Extract features from RR interval
            rr_features = extract_rr_features(filtered_rr_data)

            # Combine features for each minute
            for minute_key in actigraphy_features.keys():
                if minute_key in rr_features.keys():  # Ensure data exists for both sensors for this minute
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
    return df


def prepare_data(df, config, train_data_path="train_data_deep.csv", test_data_path="test_data_deep.csv"):
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        # Split users into 70-30 ensuring balance of y
        user_classes = df.groupby('user_id')['y'].mean()

        force_test_users = config.get('force.test.users').data
        if force_test_users == '':
            train_users, test_users = train_test_split(user_classes.index, test_size=0.3, stratify=user_classes)
        else:
            force_test_users_value = [int(c) for c in force_test_users.split(',')]
            test_users = pd.Index(force_test_users_value, name='user_id')
            train_users = pd.Index(user_classes[~user_classes.index.isin(test_users)].index, name='user_id')

        train_data = df[df['user_id'].isin(train_users)]
        test_data = df[df['user_id'].isin(test_users)]

        # Drop unnecessary features
        x_train = train_data.drop(columns=['user_id', 'day', 'hour', 'minute', 'y'])
        y_train = train_data['y']
        x_test = test_data.drop(columns=['user_id', 'day', 'hour', 'minute', 'y'])
        y_test = test_data['y']
        y_test_user_id = test_data['user_id']

        # Standardize data
        scaler = StandardScaler()
        x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
        x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

        # Feature selection
        feature_selection_algorithm_to_run = config.get('feature.selection').data
        selected_train_features = perform_feature_selection_method(x_train_scaled, y_train,
                                                                   feature_selection_algorithm_to_run)

        # Save filtered train and test data to CSV
        train_data = x_train_scaled[selected_train_features.tolist()]
        train_data = train_data.assign(y=y_train.values)
        train_data.to_csv(train_data_path, index=False)

        test_data = x_test_scaled
        test_data = test_data.assign(y=y_test.values)
        test_data = test_data.assign(user_id=y_test_user_id.values)
        test_data.to_csv(test_data_path, index=False)
    else:
        print("The train and test data files already exist. No need to run the code again.")
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        x_train = train_data.drop(columns=['y'])
        y_train = train_data['y']
        x_test = test_data.drop(columns=['user_id', 'y'])
        y_test = test_data['y']

        selected_train_features = pd.read_csv(train_data_path, usecols=lambda column: column != 'y', nrows=0)
        selected_train_features = selected_train_features.columns

    return train_data, test_data, x_train, y_train, x_test, y_test, selected_train_features


def create_and_save_bayesian_network(description_file='dataset_description.json',
                                     dataset_file='train_data_deep.csv',
                                     epsilon=0, degree_of_bayesian_network=2):
    if not os.path.exists(description_file):
        describer = DataDescriber()
        # Describe the dataset to create a Bayesian network
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file=dataset_file,
                                                                epsilon=epsilon,
                                                                k=degree_of_bayesian_network)
        # Save dataset description to a JSON file
        describer.save_dataset_description_to_file(description_file)
        print("Bayesian network description saved.")
        # Optionally display the Bayesian network
        # display_bayesian_network(describer.bayesian_network)
    else:
        print(f"The {description_file} file already exists. No need to run the code.")


def generate_and_save_synthetic_data(synthetic_data_file='synthetic_data.csv',
                                     description_file='dataset_description.json',
                                     num_tuples_to_generate=100000):
    if not os.path.exists(synthetic_data_file):
        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
        # Save synthetic data to a CSV file
        generator.save_synthetic_data(synthetic_data_file)
        print("Synthetic data generated and saved.")
    else:
        print(f"The {synthetic_data_file} file already exists. No need to run the code.")


def balance_data(df, target_column):
    """
    Balances the dataset to have an equal number of samples for each class.
    """
    counts = df[target_column].value_counts()
    min_count = counts.min()
    balanced_sample = df.groupby(target_column).apply(lambda x: x.sample(min_count)).reset_index(drop=True)
    return balanced_sample


if __name__ == "__main__":
    main()
