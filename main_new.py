import os
import pandas as pd
from feature_extraction import extract_actigraphy_features, extract_rr_features, process_rr_data, \
    filter_data_by_sleep_intervals
from classifiers import run_classifiers, run_classifiers_after_deep, write_results_to_csv, run_regressors_after_deep, \
    write_results_to_csv_regressor
from feature_selection import perform_feature_selection_method
from config_loader import ConfigLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import get_column_plot
from sdv.evaluation.single_table import evaluate_quality
from oversampling import perform_oversampling_deep_method, get_deep_model_name, get_basic_method_name
import plotly.io as pio
import time


def main():
    config = ConfigLoader.get_instance()
    write_class_value = config.get('write.class').data
    write_classes = [int(c) for c in write_class_value.split(',')]
    oversampling_deep_algorithm_to_run = config.get('oversampling.deep.algorithm').data
    oversampling_algorithm_to_run = config.get('oversampling.algorithm').data
    num_tuples_to_generate = int(config.get('num.tuples.to.generate').data)

    df = process_user_folders_new()
    train_data_for_deep, test_data, x_train, y_train, x_test, y_test, selected_train_features = prepare_data_new(df,
                                                                                                                     config)
    if oversampling_deep_algorithm_to_run != '0':
        start_time = time.time()
        perform_oversampling_deep_method(train_data_for_deep, oversampling_deep_algorithm_to_run,
                                         num_tuples_to_generate)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

    
    process_rr_data(train_data_for_deep, 'synthetic_data.csv')
    process_rr_data(test_data, 'test_data_deep.csv')

    if os.path.exists("synthetic_data.csv"):
        df_after_deep = pd.read_csv('synthetic_data.csv')
        for column in train_data_for_deep.columns:
            if column != 'y':
                plot_feature_comparison(train_data_for_deep, df_after_deep, column)

        evaluate_and_save_quality_details(train_data_for_deep, df_after_deep)

        # bilancio i dati sintetici se non lo fossero già
        balanced_sample = balance_data(df_after_deep, 'y')

        # lancio i classificatori, passando i dati sintetici come train set,
        # mentre uso il 30 lasciato all'inizio come test set
        x_train_after_deep = balanced_sample.drop(columns=['y'])
        y_train_after_deep = balanced_sample['y']

    test_data_user_id = test_data.user_id
    test_data = test_data[['y'] + selected_train_features.tolist()]
    df_test = test_data.groupby(test_data_user_id)

    '''
    #regression
    results_after_deep = run_regressors_after_deep(x_train_after_deep, y_train_after_deep, df_test)
    write_results_to_csv_regressor("output_synthetic.csv", results_after_deep)
    
    results_after_deep = run_regressors_after_deep(x_train, y_train, df_test)
    write_results_to_csv_regressor("output_no_synthetic.csv", results_after_deep)
    '''

    results_after_deep = run_classifiers_after_deep(x_train_after_deep, y_train_after_deep, df_test)
    write_results_to_csv(f"output_synthetic_{get_deep_model_name(oversampling_deep_algorithm_to_run)}.csv",
                         results_after_deep, write_classes)

    # lancio i classificatori passando il 70% iniziale come train set
    # mentre uso il 30 lasciato all'inizio come test set
    results_after_deep = run_classifiers_after_deep(x_train, y_train, df_test)
    write_results_to_csv(f"output_no_synthetic_{get_deep_model_name(oversampling_deep_algorithm_to_run)}.csv",
                         results_after_deep, write_classes)

    '''
    deccomentare per lanciare i metodi standard in modalità 70-30
    x_train_scaled, y_train = perform_oversampling_method(x_train, y_train, oversampling_algorithm_to_run)
    results_after_deep = run_classifiers_after_deep(x_train_scaled, y_train, df_test)
    write_results_to_csv(f"output_no_synthetic_{get_deep_model_name(oversampling_deep_algorithm_to_run)}.csv",
                         results_after_deep, write_classes)
    

    # lancio i classificatori senza operazioni di deep learning
    users_df = df.groupby(df.user_id)
    # Run classifiers on the combined data
    results = run_classifiers(users_df)
    write_results_to_csv(f"output_basic_oversampled_{get_basic_method_name(oversampling_algorithm_to_run)}.csv",
                         results, write_classes)
    '''


def evaluate_chatgpt_data():
    df_chatgpt = pd.read_csv('syntethic_data_chatgpt.csv')
    train_data_for_deep = pd.read_csv('train_data_deep.csv')
    for column in train_data_for_deep.columns:
        if column != 'y':
            plot_feature_comparison(train_data_for_deep, df_chatgpt, column)
    evaluate_and_save_quality_details(train_data_for_deep, df_chatgpt)


def check_and_process_combined_features():
    """
    Processes combined features and applying necessary transformations.

    Returns
    -------
    DataFrame
        A pandas DataFrame that has been processed and is ready for further analysis.
    """
    # Check if the combined_features.csv file exists in the current directory
    if not os.path.exists("combined_features.csv"):
        print("Processing features...")
        return process_user_folders()
    else:
        print("The combined_features.csv file already exists. No need to run the code.")
        return pd.read_csv('combined_features.csv')


def process_user_folders():
    """
    Processes user folders in the current directory, extracts features from actigraphy and RR data, and combines these
    features into a single CSV file.

    This function loops through each folder that starts with "user_" in the current working directory, extracts
    user-specific questionnaire data, filters actigraphy and RR data based on sleep intervals, extracts relevant features,
    and combines them into a comprehensive DataFrame.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing combined features from actigraphy and RR data for each minute where both sets
        of data are available. The DataFrame is also written to 'combined_features.csv' in the current directory.

    Side Effects
    ------------
    1. Writes a CSV file 'combined_features.csv' in the current working directory containing the combined features
    for all users processed.
    """
    current_directory = os.getcwd()
    all_combined_features = []  # List to store combined features for all users
    all_users_actigraph_data = []
    all_users_rr_data = []

    # Loop through folders for each user
    for user_folder in os.listdir(current_directory):
        if user_folder.startswith("user_") and os.path.isdir(user_folder):
            # Extract user ID from folder name
            user_id = int(user_folder.split("_")[1])

            # Load questionnaire data for the user
            questionnaire_data = pd.read_csv(os.path.join(current_directory, user_folder, "questionnaire.csv"))

            # Encode labels based on PSQI score
            if True:
                y = (questionnaire_data['Pittsburgh'] <= 5).astype(int).to_string(index=False)
            else:
                y = questionnaire_data['Daily_stress']

            # Filter actigraphy and RR data by sleep intervals
            filtered_actigraphy_data, filtered_rr_data = filter_data_by_sleep_intervals(user_folder,
                                                                                        current_directory)

            for index, row in filtered_actigraphy_data.iterrows():
                data = {'user_id': user_id, 'y': y}
                data.update({k: v for k, v in row.to_dict().items() if k not in ['user_id', 'y']})
                all_users_actigraph_data.append(data)

            for index, row in filtered_rr_data.iterrows():
                data = {'user_id': user_id, 'y': y}
                data.update({k: v for k, v in row.to_dict().items() if k not in ['user_id', 'y']})
                all_users_rr_data.append(data)

            # Extract features from actigraphy data
            actigraphy_features = extract_actigraphy_features(filtered_actigraphy_data)

            # Extract features from RR interval
            rr_features = extract_rr_features(filtered_rr_data)

            # Combine features for each minute
            for minute_key in rr_features.keys():
                if minute_key in rr_features.keys():  # Ensure data exists for both sensors for this minute
                    minute_dict = {
                        'user_id': user_id,
                        'day': minute_key[0],
                        'hour': minute_key[1],
                        'minute': minute_key[2],
                        # **actigraphy_features[minute_key],  # Add all actigraphy features
                        **rr_features[minute_key],  # Add all RR interval features
                        'y': int(y)
                    }
                    all_combined_features.append(minute_dict)

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(all_combined_features)

    # Write DataFrame to a CSV file
    df.to_csv('combined_features.csv', index=False)

    df_actigraph_data = pd.DataFrame(all_users_actigraph_data)
    actigraph_columns = ['user_id', 'Axis1', 'Axis2', 'Axis3', 'Steps', 'HR', 'Inclinometer Off',
                         'Inclinometer Standing',
                         'Inclinometer Sitting', 'Inclinometer Lying', 'Vector Magnitude', 'day', 'time', 'y']
    df_actigraph_data[actigraph_columns].to_csv('all_users_actigraph_data.csv', index=False)

    df_rr_data = pd.DataFrame(all_users_rr_data)
    rr_columns = ['user_id', 'ibi_s', 'day', 'time', 'y']
    df_rr_data[rr_columns].to_csv('all_users_rr_data.csv', index=False)

    print("Features processed!")
    return df


def prepare_data(df, config, train_data_path="train_data_deep.csv", test_data_path="test_data_deep.csv"):
    """
    Prepares the data for model training by splitting into training and test datasets and applying feature selection.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to process.
    config : ConfigLoader
        The configuration settings loaded via ConfigLoader.

    Returns
    -------
    tuple
        A tuple containing training data, test data, training feature matrix, training labels, test feature matrix, test labels, and selected features.
    """
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
        x_train = train_data.drop(columns=['user_id', 'day', 'hour', 'minute', 'y', 'time'], errors='ignore')
        y_train = train_data['y']
        x_test = test_data.drop(columns=['user_id', 'day', 'hour', 'minute', 'y', 'time'], errors='ignore')
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


def balance_data(df, target_column):
    """
    Balances the dataset to have an equal number of samples for each class in the specified target column.

    Parameters
    ----------
    df : DataFrame
        The pandas DataFrame to balance.
    target_column : str
        The name of the column in df that contains the class labels.

    Returns
    -------
    DataFrame
        A balanced DataFrame where each class has the same number of samples.
    """
    counts = df[target_column].value_counts()
    min_count = counts.min()
    balanced_sample = df.groupby(target_column).apply(lambda x: x.sample(min_count)).reset_index(drop=True)
    shuffled_balanced_sample = balanced_sample.sample(frac=1).reset_index(drop=True)
    return shuffled_balanced_sample


def plot_feature_comparison(real_data, synthetic_data, feature_name, output_path="./iframe_figures/"):
    """
    Generates a comparison plot for a specific feature between real and synthetic data sets, and saves it as an HTML file.

    Parameters
    ----------
    real_data : DataFrame
        The pandas DataFrame containing the real data.
    synthetic_data : DataFrame
        The pandas DataFrame containing the synthetic data.
    feature_name : str
        The name of the feature to compare.
    output_path : str, optional
        The file path where the output HTML file will be saved. Default is "./iframe_figures/".

    Returns
    -------
    None
        The function saves the plot as an HTML file and does not return any value.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)

    fig = get_column_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
        column_name=feature_name
    )

    filename = f"{output_path}{feature_name}_comparison_plot.html"
    pio.write_html(fig, file=filename)


def evaluate_and_save_quality_details(real_data, synthetic_data, output_path="./iframe_figures/"):
    """
    Evaluates the quality of synthetic data compared to real data using statistical metrics and saves the details to CSV files.

    Parameters
    ----------
    real_data : DataFrame
        The pandas DataFrame containing the real data.
    synthetic_data : DataFrame
        The pandas DataFrame containing the synthetic data.

    Returns
    -------
    None
        The function saves the evaluation details as CSV files and does not return any value.
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)

    quality_report = evaluate_quality(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata)

    shapes_details = quality_report.get_details(property_name='Column Shapes')
    trends_details = quality_report.get_details(property_name='Column Pair Trends')

    shapes_details.to_csv('column_shapes_details.csv', index=False)
    trends_details.to_csv('column_pair_trends_details.csv', index=False)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fig = quality_report.get_visualization(property_name='Column Shapes')
    filename = f"{output_path}report_column_shapes.html"
    pio.write_html(fig, file=filename)

    fig = quality_report.get_visualization(property_name='Column Pair Trends')
    filename = f"{output_path}report_column_pair_trends.html"
    pio.write_html(fig, file=filename)


def process_user_folders_new():
    current_directory = os.getcwd()
    all_users_rr_data = []

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
            for index, row in filtered_rr_data.iterrows():
                data = {'user_id': user_id, 'y': y}
                data.update({k: v for k, v in row.to_dict().items() if k not in ['user_id', 'y']})
                all_users_rr_data.append(data)

    df = pd.DataFrame(all_users_rr_data)

    config = ConfigLoader.get_instance()
    force_test_users = config.get('force.test.users').data
    force_test_users_value = [int(c) for c in force_test_users.split(',')]
    user_classes = df.groupby('user_id')['y'].mean()
    test_users = pd.Index(force_test_users_value, name='user_id')
    train_users = pd.Index(user_classes[~user_classes.index.isin(test_users)].index, name='user_id')

    train_data = df[df['user_id'].isin(train_users)]
    test_data = df[df['user_id'].isin(test_users)]
    rr_columns = ['user_id', 'ibi_s', 'day', 'time', 'y']
    train_data[rr_columns].to_csv('all_users_rr_data_train.csv', index=False)
    test_data[rr_columns].to_csv('all_users_rr_data_test.csv', index=False)

    print("Features processed!")
    return df


def prepare_data_new(df, config, train_data_path="all_users_rr_data_train.csv",
                     test_data_path="all_users_rr_data_test.csv"):
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    # Drop unnecessary features
    x_train = train_data.drop(columns=['user_id', 'day', 'hour', 'minute', 'y', 'time'], errors='ignore')
    y_train = train_data['y']
    x_test = test_data.drop(columns=['user_id', 'day', 'hour', 'minute', 'y', 'time'], errors='ignore')
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

    x_train = train_data.drop(columns=['y'])
    y_train = train_data['y']
    x_test = test_data.drop(columns=['user_id', 'y'])
    y_test = test_data['y']

    selected_train_features = pd.read_csv(train_data_path, usecols=lambda column: column != 'y', nrows=0)
    selected_train_features = selected_train_features.columns

    return train_data, test_data, x_train, y_train, x_test, y_test, selected_train_features


if __name__ == "__main__":
    main()
