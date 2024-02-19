import os
import pandas as pd
from feature_extraction import extract_actigraphy_features, extract_rr_features
import json

current_directory = os.getcwd()

all_combined_features = []  # Lista per memorizzare le features combinate di tutti gli utenti

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
        
        # Combinazione diretta delle caratteristiche per ogni minuto
        for minute_key in actigraphy_features.keys():
            if minute_key in rr_features.keys():  # Assicurati che ci siano dati per entrambi i sensori per questo minuto
                minute_dict = {
                    'user_id': user_id,
                    'day': minute_key[0],
                    'hour': minute_key[1],
                    'minute': minute_key[2],
                    **actigraphy_features[minute_key],  # Aggiungi tutte le caratteristiche dell'actigrafia
                    **rr_features[minute_key],  # Aggiungi tutte le caratteristiche dell'intervallo RR
                    'y': y
                }
                all_combined_features.append(minute_dict)

stringa = ' '.join([str(dizionario) for dizionario in all_combined_features])

# Apertura del file in modalità scrittura ("w" indica che il file sarà sovrascritto se esiste già)
#with open("result.txt", "w") as file:
    # Scrivere i dati nel file
    #file.write(str(stringa))
    
# Converte la lista di dizionari in un DataFrame pandas
df = pd.DataFrame(all_combined_features)

# Scrive il DataFrame su un file CSV
df.to_csv('combined_features.csv', index=False)    
