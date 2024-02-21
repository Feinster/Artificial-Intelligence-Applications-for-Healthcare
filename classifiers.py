from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

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
    
    return accuracy, precision, recall, f1

# Function to run classifiers
def run_classifiers(X, y, n_folds):
    # Standardize input data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize classifiers
    classifiers = {
        'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier()
    }
    
    print("n_folds:", n_folds)

    # Evaluate each classifier with cross-validation
    results = {}
    for clf_name, clf in classifiers.items():
        # Evaluate classifier using cross-validation
        cv_scores = cross_val_score(clf, X_scaled, y, cv=n_folds, scoring='accuracy')
        results[clf_name] = {'Cross-Validation Accuracy': cv_scores.mean()}
    
    return results
