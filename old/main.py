import os
import pandas as pd
from feature_extraction import extract_actigraphy_features, extract_rr_features
from feature_selection import (rfe_feature_selection, 
                                random_forest_feature_selection, 
                                extra_trees_feature_selection, 
                                information_gain_feature_selection, 
                                variance_threshold_feature_selection)
from classifiers import run_classifiers

current_directory = os.getcwd()

with open("result.txt", "w") as result_file:

    user_features = {}  # Dictionary per memorizzare le features di ogni utente
    user_labels = {}  # Dictionary per memorizzare le labels di ogni utente

    # Loop attraverso le cartelle di ogni utente
    for user_folder in os.listdir(current_directory):
        if user_folder.startswith("user_") and os.path.isdir(user_folder):
            # Estrae l'ID dell'utente dal nome della cartella
            user_id = int(user_folder.split("_")[1])

            # Carica i dati del questionario per l'utente
            questionnaire_data = pd.read_csv(os.path.join(current_directory, user_folder, "questionnaire.csv"))

            # Codifica le labels in base al punteggio PSQI
            y = (questionnaire_data['Pittsburgh'] <= 5).astype(int).to_string(index=False)

            # Legge i dati dell'actigraphy e dell'intervallo RR
            actigraphy_data = pd.read_csv(os.path.join(current_directory, user_folder, "Actigraph.csv"))
            rr_data = pd.read_csv(os.path.join(current_directory, user_folder, "RR.csv"))

            # Estrae le features dai dati dell'actigraphy
            actigraphy_features = extract_actigraphy_features(actigraphy_data)

            # Estrae le features dall'intervallo RR
            rr_features = extract_rr_features(rr_data)

            combined_features = {**actigraphy_features, **rr_features}

            # Aggiungi la colonna 'y' a tutte le righe per quell'utente
            for minute_key, minute_features in combined_features.items():
                minute_features['y'] = y
                
            # Stampa le righe per l'utente
            print(f"User {user_id} features:")
            for minute_key, minute_features in combined_features.items():
                print(minute_key, minute_features)

    # Scrive le features per ogni utente in ordine crescente sul file "result.txt"
    #for user_id, features in sorted(user_features.items()):
    #    print(f"User {user_id} features:", features)
    #    print("\n")

    # Crea DataFrame dalle features combinate
    #X = pd.DataFrame(user_features.values())  

    # Esegue la selezione delle features ed esegue il classificatore sulle feature selezionate
    #*************************************** Recursive Feature Elimination ***************************************
    #selected_features_rfe = rfe_feature_selection(X, y, n_features_to_select=10)
    #
    #print("Selected features after Recursive Feature Elimination (RFE):", file=result_file)
    #print(selected_features_rfe, file=result_file)
    #print("\n", file=result_file)
    #
    #X_selected_rfe = X[selected_features_rfe]
    #
    #results_rfe = run_classifiers(X_selected_rfe, y)
    #
    #print("Results after Recursive Feature Elimination (RFE):", file=result_file)
    #print(results_rfe, file=result_file)
    #print("\n", file=result_file)
    #
    ##*************************************** Random Forest Importance ***************************************
    #selected_features_rf = random_forest_feature_selection(X, y)
    #
    #print("Selected features after Random Forest Importance:", file=result_file)
    #print(selected_features_rf, file=result_file)
    #print("\n", file=result_file)
    #
    #X_selected_rf = X[selected_features_rf]
    #
    #results_rf = run_classifiers(X_selected_rf, y)
    #
    #print("Results after Random Forest Importance:", file=result_file)
    #print(results_rf, file=result_file)
    #print("\n", file=result_file)
    #
    ##*************************************** Extra Trees Classifier ***************************************
    #selected_features_et = extra_trees_feature_selection(X, y)
    #
    #print("Selected features after Extra Trees Classifier:", file=result_file)
    #print(selected_features_et, file=result_file)
    #print("\n", file=result_file)
    #
    #X_selected_et = X[selected_features_et]
    #
    #results_et = run_classifiers(X_selected_et, y)
    #
    #print("Results after Extra Trees Classifier:", file=result_file)
    #print(results_et, file=result_file)
    #print("\n", file=result_file)
    #
    ##*************************************** Information Gain ***************************************
    #selected_features_mi = information_gain_feature_selection(X, y)
    #
    #print("Selected features after Information Gain:", file=result_file)
    #print(selected_features_mi, file=result_file)
    #print("\n", file=result_file)
    #
    #X_selected_mi = X[selected_features_mi]
    #
    #results_mi = run_classifiers(X_selected_mi, y)
    #
    #print("Results after Information Gain:", file=result_file)
    #print(results_mi, file=result_file)
    #print("\n", file=result_file)
    #
    ##*************************************** Variance Threshold Method ***************************************
    #selected_features_vt = variance_threshold_feature_selection(X, threshold=0.1)
    #
    #print("Selected features after Variance Threshold Method:", file=result_file)
    #print(selected_features_vt, file=result_file)
    #print("\n", file=result_file)
    #
    #X_selected_vt = X[selected_features_vt]
    #
    #results_vt = run_classifiers(X_selected_vt, y)
    #
    #print("Results after Variance Threshold Method:", file=result_file)
    #print(results_vt, file=result_file)
    #print("\n", file=result_file)
