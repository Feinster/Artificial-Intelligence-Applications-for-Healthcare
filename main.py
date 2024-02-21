import os
import pandas as pd
from feature_extraction import extract_actigraphy_features, extract_rr_features
import json
from datetime import datetime

current_directory = os.getcwd()

all_combined_features = []  # Lista per memorizzare le features combinate di tutti gli utenti

# Carica i dati dai file Actigraph.csv e RR.csv e restituisci i dati filtrati in base agli intervalli di sonno
def filter_data_by_sleep_intervals(user_folder):
    # Carica i dati dal file sleep.csv per ottenere gli intervalli di sonno per questo utente
    sleep_data = pd.read_csv(os.path.join(current_directory, user_folder, "sleep.csv"))

    # Carica i dati dai file Actigraph.csv e RR.csv
    actigraphy_data = pd.read_csv(os.path.join(current_directory, user_folder, "Actigraph.csv"))
    rr_data = pd.read_csv(os.path.join(current_directory, user_folder, "RR.csv"))

    # Inizializza una lista per memorizzare i dati filtrati
    filtered_actigraphy_data = []
    filtered_rr_data = []

    # Itera attraverso le righe di sleep_data per ottenere gli intervalli di sonno
    for _, row in sleep_data.iterrows():
        in_bed_date = row["In Bed Date"]
        in_bed_time = row["In Bed Time"]
        out_bed_date = row["Out Bed Date"]
        out_bed_time = row["Out Bed Time"]
        
        if((in_bed_date != out_bed_date) &
        (
                        (column_data == in_bed_date) &  # I dati sono nel giorno dell'inizio del periodo di sonno
                        (column_time >= in_bed_time)  # Dati dopo l'inizio del periodo di sonno
                    ) | 
                    (
                        (column_data == out_bed_date) &  # I dati sono nel giorno della fine del periodo di sonno
                        (column_time <= out_bed_time)  # Dati prima della fine del periodo di sonno
                    )):
        
        filtered_actigraphy_data.append(
            actigraphy_data[
                (
                    (actigraphy_data['day'] == in_bed_date) &  # I dati sono nello stesso giorno dell'inizio del periodo di sonno
                    (actigraphy_data['time'] >= in_bed_time) &  # Dati dopo l'inizio del periodo di sonno
                    (actigraphy_data['time'] <= out_bed_time)    # Dati prima della fine del periodo di sonno
                ) | 
                (
                    (in_bed_date != out_bed_date) &  # Il periodo di sonno attraversa la mezzanotte
                    (
                        (
                            (actigraphy_data['day'] == in_bed_date) &  # I dati sono nel giorno dell'inizio del periodo di sonno
                            (actigraphy_data['time'] >= in_bed_time)  # Dati dopo l'inizio del periodo di sonno
                        ) | 
                        (
                            (actigraphy_data['day'] == out_bed_date) &  # I dati sono nel giorno della fine del periodo di sonno
                            (actigraphy_data['time'] <= out_bed_time)  # Dati prima della fine del periodo di sonno
                        )
                    )
                )
            ]
        )

        filtered_rr_data.append(
            rr_data[
                (
                    (rr_data['day'] == in_bed_date) &  # I dati sono nello stesso giorno dell'inizio del periodo di sonno
                    (rr_data['time'] >= in_bed_time) &  # Dati dopo l'inizio del periodo di sonno
                    (rr_data['time'] <= out_bed_time)    # Dati prima della fine del periodo di sonno
                ) | 
                (
                    (in_bed_date != out_bed_date) &  # Il periodo di sonno attraversa la mezzanotte
                    (
                        (
                            (rr_data['day'] == in_bed_date) &  # I dati sono nel giorno dell'inizio del periodo di sonno
                            (rr_data['time'] >= in_bed_time)  # Dati dopo l'inizio del periodo di sonno
                        ) | 
                        (
                            (rr_data['day'] == out_bed_date) &  # I dati sono nel giorno della fine del periodo di sonno
                            (rr_data['time'] <= out_bed_time)  # Dati prima della fine del periodo di sonno
                        )
                    )
                )
            ]
        )

    # Concatena i dati filtrati in un unico DataFrame per ciascun sensore
    filtered_actigraphy_data = pd.concat(filtered_actigraphy_data)
    filtered_rr_data = pd.concat(filtered_rr_data)

    return filtered_actigraphy_data, filtered_rr_data

# Loop attraverso le cartelle di ogni utente
for user_folder in os.listdir(current_directory):
    if user_folder.startswith("user_") and os.path.isdir(user_folder):
        # Estrae l'ID dell'utente dal nome della cartella
        user_id = int(user_folder.split("_")[1])

        # Carica i dati del questionario per l'utente
        questionnaire_data = pd.read_csv(os.path.join(current_directory, user_folder, "questionnaire.csv"))

        # Codifica le labels in base al punteggio PSQI
        y = (questionnaire_data['Pittsburgh'] <= 5).astype(int).to_string(index=False)

        filtered_actigraphy_data, filtered_rr_data = filter_data_by_sleep_intervals(user_folder)
        
        # Estrae le features dai dati dell'actigraphy
        actigraphy_features = extract_actigraphy_features(filtered_actigraphy_data)

        # Estrae le features dall'intervallo RR
        rr_features = extract_rr_features(filtered_rr_data)
        
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
    
# Converte la lista di dizionari in un DataFrame pandas
df = pd.DataFrame(all_combined_features)

# Scrive il DataFrame su un file CSV
df.to_csv('combined_features.csv', index=False)    
