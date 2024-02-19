from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

def evaluate_classifier(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1

def run_classifiers(X, y):
    # Standardizza i dati in ingresso
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dei dati in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Inizializza i classificatori
    classifiers = {
        'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier()
    }
    
    # Valuta ogni classificatore
    results = {}
    for clf_name, clf in classifiers.items():
        accuracy, precision, recall, f1 = evaluate_classifier(clf, X_train, X_test, y_train, y_test)
        results[clf_name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    
    return results
