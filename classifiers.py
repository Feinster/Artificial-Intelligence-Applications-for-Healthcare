from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from oversampling import perform_oversampling_method, perform_safe_level_smote
from config_loader import ConfigLoader
import csv
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.svm import SVC
import time

def evaluate_classifier(clf, x_train, x_test, y_train, y_test, do_fit):
    """
    Evaluates a classifier on given training and test datasets and calculates performance metrics.

    Parameters
    ----------
    clf : Classifier
        The classifier instance to evaluate.
    x_train : DataFrame
        The training feature dataset.
    x_test : DataFrame
        The test feature dataset.
    y_train : Series
        The training labels.
    y_test : Series
        The test labels.
    do_fit : bool
        Flag to determine whether to fit the classifier with training data before prediction.

    Returns
    -------
    tuple
        A tuple containing balanced accuracy, precision, recall, and F1 score of the classifier.
    """
    # Train the classifier
    if do_fit:
        clf.fit(x_train, y_train)

    # Make predictions
    y_pred = clf.predict(x_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # print(f'Modello: {clf}, Accuracy: {accuracy}, Precision:{precision}, Recall:{recall}, F1-score:{f1}')

    return accuracy, balanced_accuracy, precision, recall, f1


def run_classifiers(users_df):
    """
    Runs multiple classifiers on a dataset, applying oversampling methods and evaluating each classifier's performance.

    Parameters
    ----------
    users_df : dict
        A dictionary grouping data by user, where each key is a user identifier and the value is their respective data.

    Returns
    -------
    dict
        A dictionary containing evaluation results for each classifier and user, indexed by classifier name,
        test group key, and last class label.
    """
    config = ConfigLoader.get_instance()
    oversampling_algorithm_to_run = config.get('oversampling.algorithm').data

    # Initialize classifiers
    classifiers = {
        'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(kernel='linear')
    }

    scaler = StandardScaler()
    results = {}
    execution_time = 0
    
    for clf_name, clf in classifiers.items():
        for test_key, test_group in users_df:

            train_data = pd.concat([group for key, group in users_df if key != test_key])

            if oversampling_algorithm_to_run == '4':
                train_data = perform_safe_level_smote(train_data, 'y')

            columns_to_drop = ['user_id', 'day', 'hour', 'minute', 'y', 'time']
            existing_columns = [col for col in columns_to_drop if col in train_data.columns]
            x_train = train_data.drop(columns=existing_columns)

            scaler.fit(x_train)
            x_train_scaled = scaler.transform(x_train)
            y_train = train_data['y']
            existing_columns = [col for col in columns_to_drop if col in test_group.columns]
            x_test = test_group.drop(columns=existing_columns)

            x_test_scaled = scaler.transform(x_test)
            y_test = test_group['y']
            y = y_test.iloc[-1]

            # Perform oversampling
            start_time = time.time()
            x_train_scaled, y_train = perform_oversampling_method(x_train_scaled, y_train,
                                                                  oversampling_algorithm_to_run)
            end_time = time.time()
            execution_time += end_time - start_time

            results[clf_name, test_key, y] = evaluate_classifier(clf, x_train_scaled, x_test_scaled, y_train, y_test,
                                                                 True)
                                                                 
    print(f"Execution time: {execution_time} seconds")
    return results


def run_classifiers_after_deep(x_train, y_train, test_data):
    """
    Evaluates multiple classifiers on test data after training with a common set of training data,
    after deep learning pre-processing for oversampling.

    Parameters
    ----------
    x_train : DataFrame
        The training feature dataset.
    y_train : Series
        The training labels.
    test_data : dict
        A dictionary grouping test data.

    Returns
    -------
    dict
        A dictionary containing evaluation results for each classifier and test group, indexed by classifier name, test group key, and last class label.
    """
    # Initialize classifiers
    classifiers = {
        'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(kernel='linear')
    }

    results = {}

    for clf_name, clf in classifiers.items():
        clf.fit(x_train, y_train)
        for test_key, test_group in test_data:
            x_test = test_group.drop(columns=['y'])
            # test_group.to_csv('test_group_{}_{}.csv'.format(clf_name, test_key), index=False)
            y_test = test_group['y']
            y = y_test.iloc[-1]
            print('Running classifier {}...'.format(clf_name))
            print('test key: {}'.format(test_key))
            results[clf_name, test_key, y] = evaluate_classifier(clf, x_train, x_test, y_train, y_test, False)

    return results


def write_results_to_csv(csv_file_path, results, write_classes):
    """
    Writes the results of classifier evaluations to a CSV file.

    Parameters
    ----------
    csv_file_path : str
        The path to the CSV file where results will be saved.
    results : dict
        A dictionary containing evaluation metrics for each classifier, indexed by model name, user ID, and class label.
    write_classes : list
        A list of class labels to include in the CSV.

    Side Effects
    ------------
    Writes to a CSV file specified by `csv_file_path`.
    """
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header row
        csv_writer.writerow(['Model', 'User_id', 'Class', 'Acc', 'Bal_Acc', 'Precision', 'Recall', 'F1-score'])

        # Write data rows
        for key, values in results.items():
            model, user_id, classe = key
            if classe in write_classes:
                csv_writer.writerow([model, user_id, classe] + list(values))


def run_regressors_after_deep(x_train, y_train, test_data):
    """
    Evaluates multiple regressors on test data after training with a common set of training data,
    after deep learning pre-processing for oversampling.

    Parameters
    ----------
    x_train : DataFrame
        The training feature dataset.
    y_train : Series
        The training labels.
    test_data : dict
        A dictionary grouping test data.

    Returns
    -------
    dict
        A dictionary containing evaluation results for each regressor and test group, indexed by regressor name, test group key.
    """
    # Initialize regressors
    regressors = {
        'AdaBoost': AdaBoostRegressor(),
        'KNN': KNeighborsRegressor(),
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor()
    }

    results = {}

    for reg_name, reg in regressors.items():
        reg.fit(x_train, y_train)
        for test_key, test_group in test_data:
            x_test = test_group.drop(columns=['y'])
            y_test = test_group['y']
            y = y_test.iloc[-1]
            y_pred = reg.predict(x_test)
            y_pred_mean = np.mean(y_pred)
            print('Running regressor {}...'.format(reg_name))
            print('test key: {}'.format(test_key))
            results[reg_name, test_key, y, y_pred_mean] = evaluate_regressor(reg, x_test, y_test)

    return results


def evaluate_regressor(reg, x_test, y_test):
    """
    Evaluates a regressor on given training and test datasets and calculates performance metrics.

    Parameters
    ----------
    reg : Regressor
        The regressor instance to evaluate.
    x_test : DataFrame
        The test feature dataset.
    y_test : Series
        The test labels.

    Returns
    -------
    tuple
        A tuple containing Mean Squared Error and R^2 score of the regressor.
    """
    y_pred = reg.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2


def write_results_to_csv_regressor(csv_file_path, results):
    """
    Writes the results of classifier evaluations to a CSV file.

    Parameters
    ----------
    csv_file_path : str
        The path to the CSV file where results will be saved.
    results : dict
        A dictionary containing evaluation metrics for each classifier, indexed by model name, user ID, and class label.

    Side Effects
    ------------
    Writes to a CSV file specified by `csv_file_path`.
    """
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header row
        csv_writer.writerow(['Model', 'User_id', 'Real_y', 'Pred_y', 'MSE', 'R2'])

        # Write data rows
        for key, values in results.items():
            model, user_id, real_y, pred_y = key
            csv_writer.writerow([model, user_id, real_y, pred_y] + list(values))
