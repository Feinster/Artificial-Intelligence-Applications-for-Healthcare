import pandas as pd
from config_loader import ConfigLoader
from oversampling import perform_oversampling_method


def perform_oversampling_and_save(input_file, output_file):
    config = ConfigLoader.get_instance()
    oversampling_algorithm_to_run = config.get('oversampling.algorithm').data

    features_df = pd.read_csv(input_file)

    y = features_df['Anxiety']
    #x = features_df.drop(
    #    columns=['In Bed Time', 'Out Bed Time', 'Onset Time', 'Latency',
    #             'Efficiency', 'Total Minutes in Bed', 'Total Sleep Time (TST)', 'Wake After Sleep Onset (WASO)',
    #             'Number of Awakenings', 'Average Awakening Length', 'Movement Index', 'Fragmentation Index',
    #             'Sleep Fragmentation Index', 'Sleep Quality'])

    #x = features_df.drop(
    #    columns=['Latency', 'Total Sleep Time (TST)', 'Fragmentation Index', 'Anxiety'])

    x = features_df.drop(
        columns=['Latency', 'Total Sleep Time (TST)', 'Fragmentation Index', 'Anxiety'])

    x_resampled, y_resampled = perform_oversampling_method(x, y,
                                                           oversampling_algorithm_to_run)

    oversampled_df = pd.DataFrame(x_resampled, columns=x.columns)
    oversampled_df['Anxiety'] = y_resampled

    oversampled_df.to_csv(output_file, index=False)


input_file = "feature_vectors_red_noSteps_noVM_classe_Ansia.csv"
output_file = "oversampled_feature_vectors_red_noSteps_noVM_classe_Ansia.csv"

perform_oversampling_and_save(input_file, output_file)
