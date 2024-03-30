from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from oversampling import perform_oversampling_method, perform_safe_level_smote, perform_gaussian_mixture_clustering
from config_loader import ConfigLoader
import csv


# Function to evaluate the classifier
def evaluate_classifier(clf, x_train, x_test, y_train, y_test):
    # Train the classifier
    #clf.fit(x_train, y_train)

    # Make predictions
    y_pred = clf.predict(x_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

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

    if oversampling_algorithm_to_run == '4':
        users_df = perform_safe_level_smote(users_df, 'y')

    for clf_name, clf in classifiers.items():
        for test_key, test_group in users_df:

            train_data = pd.concat([group for key, group in users_df if key != test_key])
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

            results[clf_name, test_key, y] = evaluate_classifier(clf, x_train_scaled, x_test_scaled, y_train, y_test)

    return results


def run_classifiers_after_deep(x_train, x_test, y_train, y_test):
    # Initialize classifiers
    classifiers = {
        'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier()
    }

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    results = {}

    for clf_name, clf in classifiers.items():
        results[clf_name] = evaluate_classifier(clf, x_train_scaled, x_test_scaled, y_train, y_test)

    return results


def run_classifiers_after_deep2(x_train, y_train, test_data):
    # Initialize classifiers
    classifiers = {
        'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier()
    }

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    results = {}

    for clf_name, clf in classifiers.items():
        clf.fit(x_train_scaled, y_train)
        for test_key, test_group in test_data:
            x_test = test_group.drop(columns=['y'])
            x_test_scaled = scaler.transform(x_test)
            y_test = test_group['y']
            y = y_test.iloc[-1]
            print('Running classifier {}...'.format(clf_name))
            print('test key: {}'.format(test_key))
            results[clf_name, test_key, y] = evaluate_classifier(clf, x_train_scaled, x_test_scaled, y_train, y_test)

    return results


def save_results_to_csv(results, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write the header row
        csv_writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score'])

        # Write the results for each classifier
        for model, metrics in results.items():
            csv_writer.writerow([model, *metrics])
