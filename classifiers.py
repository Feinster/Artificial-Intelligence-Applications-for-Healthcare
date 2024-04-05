from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from oversampling import perform_oversampling_method, perform_safe_level_smote, perform_gaussian_mixture_clustering
from config_loader import ConfigLoader
import csv


# Function to evaluate the classifier
def evaluate_classifier(clf, x_train, x_test, y_train, y_test, do_fit):
    # Train the classifier
    if do_fit:
        clf.fit(x_train, y_train)

    # Make predictions
    y_pred = clf.predict(x_test)

    # Calculate evaluation metrics
    #accuracy = accuracy_score(y_test, y_pred)
    accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # print(f'Modello: {clf}, Accuracy: {accuracy}, Precision:{precision}, Recall:{recall}, F1-score:{f1}')

    return accuracy, precision, recall, f1


# Function to run classifiers
def run_classifiers(users_df):
    config = ConfigLoader.get_instance()
    oversampling_algorithm_to_run = config.get('oversampling.algorithm').data

    # Initialize classifiers
    classifiers = {
        'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier()
    }

    scaler = StandardScaler()
    results = {}

    for clf_name, clf in classifiers.items():
        for test_key, test_group in users_df:

            train_data = pd.concat([group for key, group in users_df if key != test_key])

            if oversampling_algorithm_to_run == '4':
                train_data = perform_safe_level_smote(train_data, 'y')

            x_train = train_data.drop(columns=['user_id', 'day', 'hour', 'minute', 'y'])

            scaler.fit(x_train)
            x_train_scaled = scaler.transform(x_train)
            y_train = train_data['y']
            x_test = test_group.drop(columns=['user_id', 'day', 'hour', 'minute', 'y'])

            if oversampling_algorithm_to_run == '8':
                cluster_centers, labels = perform_gaussian_mixture_clustering(x_test)

            x_test_scaled = scaler.transform(x_test)
            y_test = test_group['y']
            y = y_test.iloc[-1]

            # Perform oversampling
            x_train_scaled, y_train = perform_oversampling_method(x_train_scaled, y_train,
                                                                  oversampling_algorithm_to_run)

            results[clf_name, test_key, y] = evaluate_classifier(clf, x_train_scaled, x_test_scaled, y_train, y_test,
                                                                 True)

    return results


def run_classifiers_after_deep(x_train, y_train, test_data):
    # Initialize classifiers
    classifiers = {
        'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier()
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
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header row
        csv_writer.writerow(['Model', 'User_id', 'Class', 'Acc', 'Precision', 'Recall', 'F1-score'])

        # Write data rows
        for key, values in results.items():
            model, user_id, classe = key
            if classe in write_classes:
                csv_writer.writerow([model, user_id, classe] + list(values))
