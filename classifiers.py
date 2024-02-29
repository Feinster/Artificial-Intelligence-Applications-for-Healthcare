from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Function to evaluate the classifier
def evaluate_classifier(clf, X_train, X_test, y_train, y_test):
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    #print(f'Modello: {clf}, Accuracy: {accuracy}, Precision:{precision}, Recall:{recall}, F1-score:{f1}')
    
    return accuracy, precision, recall, f1

# Function to run classifiers
def run_classifiers(users_df):

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
            X_train = train_data.drop(columns=['user_id', 'day', 'hour', 'minute', 'y'])
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            y_train = train_data['y']
            X_test = test_group.drop(columns=['user_id', 'day', 'hour', 'minute', 'y'])
            X_test_scaled = scaler.transform(X_test)
            y_test = test_group['y']
            y = y_test.iloc[-1]

            results[clf_name, test_key, y] = evaluate_classifier(clf, X_train_scaled, X_test_scaled, y_train, y_test)

    return results