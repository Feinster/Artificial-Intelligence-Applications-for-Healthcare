import os
import pandas as pd
from feature_extraction import extract_actigraphy_features, extract_rr_features, filter_data_by_sleep_intervals
from classifiers import run_classifiers, run_classifiers_after_deep, write_results_to_csv
import csv
from feature_selection import perform_feature_selection_method
from config_loader import ConfigLoader
import pgmpy.estimators as estimators
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from sklearn.preprocessing import StandardScaler
import pickle
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.lib.utils import display_bayesian_network
from sklearn.model_selection import train_test_split
import json


def main():
    config = ConfigLoader.get_instance()
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
                filtered_actigraphy_data, filtered_rr_data = filter_data_by_sleep_intervals(user_folder,
                                                                                            current_directory)

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

    if not os.path.exists("train_data_deep.csv") or not os.path.exists("test_data_deep.csv"):
        # divido gli utenti in 70-30 garantendo il bilanciamento della y
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

        # elimino feature inutili a mano
        x_train = train_data.drop(columns=['user_id', 'day', 'hour', 'minute', 'y'])
        y_train = train_data['y']
        x_test = test_data.drop(columns=['user_id', 'day', 'hour', 'minute', 'y'])
        y_test = test_data['y']
        y_test_user_id = test_data['user_id']

        # faccio da subito la standardizzazione dati così non la faccio più dopo
        scaler = StandardScaler()
        x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
        x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

        # Feature selection
        feature_selection_algorithm_to_run = config.get('feature.selection').data

        selected_train_features = perform_feature_selection_method(x_train_scaled, y_train,
                                                                   feature_selection_algorithm_to_run)
        # in base alle feature migliori filtro il train e il test e li salvo su csv
        train_data = x_train_scaled[selected_train_features.tolist()]
        train_data = train_data.assign(y=y_train.values)
        train_data.to_csv('train_data_deep.csv', index=False)

        test_data = x_test_scaled
        test_data = test_data.assign(y=y_test.values)
        test_data = test_data.assign(user_id=y_test_user_id.values)
        test_data.to_csv('test_data_deep.csv', index=False)
    else:
        print("The train_data_deep.csv and test_data_deep.csv files already exists. No need to run the code.")
        train_data = pd.read_csv("train_data_deep.csv")
        test_data = pd.read_csv("test_data_deep.csv")
        x_train = train_data.drop(columns=['y'])
        y_train = train_data['y']
        x_test = test_data.drop(columns=['user_id', 'y'])
        y_test = test_data['y']

        selected_train_features = pd.read_csv("train_data_deep.csv", usecols=lambda column: column != 'y', nrows=0)
        selected_train_features = selected_train_features.columns

    ''' PROVA pgmpy deep method
    scaler = StandardScaler()
    scaler.fit(df)
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    
    if not os.path.exists("dag.pkl"):
        dag = estimators.HillClimbSearch(data=df_scaled).estimate(scoring_method='k2score', max_indegree=4, max_iter=10,
                                                                  show_progress=True)
        with open('dag.pkl', 'wb') as f:
            pickle.dump(dag, f)
    else:
        with open('dag.pkl', 'rb') as f:
            dag = pickle.load(f)
    
    network = BayesianNetwork(dag)
    network.fit(df_scaled)
    model = BayesianModelSampling(network)
    model.forward_sample(50)
    FINE PROVA pgmpy deep method'''

    # creo la rete bayesiana che descrive il training set e lo salvo su .json
    description_file = 'dataset_description.json'
    if not os.path.exists(description_file):
        epsilon = 0
        degree_of_bayesian_network = 2
        describer = DataDescriber()
        # Describe the dataset to create a Bayesian network
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file='train_data_deep.csv',
                                                                epsilon=epsilon,
                                                                k=degree_of_bayesian_network)

        # Save dataset description to a JSON file
        describer.save_dataset_description_to_file(description_file)
        # Display the Bayesian network
        # display_bayesian_network(describer.bayesian_network)
    else:
        print("The dataset_description.json file already exists. No need to run the code.")

    # creo i dati sintetici e li salvo su file .csv
    synthetic_data_file = 'synthetic_data.csv'
    if not os.path.exists(synthetic_data_file):
        num_tuples_to_generate = 100000
        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
        # Save synthetic data to a CSV file
        generator.save_synthetic_data(synthetic_data_file)
    else:
        print("The synthetic_data.csv file already exists. No need to run the code.")

    # lancio i classificatori, passando i dati sintetici come train set,
    # mentre uso il 30 lasciato all'inizio come test set
    df_after_deep = pd.read_csv(synthetic_data_file)

    # equilibrio i dati sintetici in modo da avere lo stesso numero di elementi per entrambe le classi
    counts = df_after_deep['y'].value_counts()
    min_count = counts.min()
    balanced_sample = df_after_deep.groupby('y').apply(lambda x: x.sample(min_count)).reset_index(drop=True)

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


if __name__ == "__main__":
    main()
