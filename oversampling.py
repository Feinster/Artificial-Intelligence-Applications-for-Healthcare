from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE
from crucio import SLS
from sklearn.cluster import MiniBatchKMeans, DBSCAN
# from clover.over_sampling import ClusterOverSampler
from sklearn.mixture import GaussianMixture
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.lib.utils import display_bayesian_network
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from sdv.single_table import CopulaGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROS = '1'
SMOTE_METHOD = '2'
BORDERLINE_SMOTE = '3'
ADASYN_METHOD = '5'
KMEANS_SMOTE = '6'
DBSCAN_SMOTE = '7'

BAYESIAN_NETWORK = '1'
CTGAN = '2'
WGAN = '3'
TVAE = '4'
COPULA_GAN = '5'
CGAN = '6'


def perform_ROS(x, y):
    ros = RandomOverSampler(random_state=0)
    x_resampled, y_resampled = ros.fit_resample(x, y)
    return x_resampled, y_resampled


def perform_smote(x, y, sampling_strategy='minority'):
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x, y)
    return x_resampled, y_resampled


def perform_borderline_smote(x, y, sampling_strategy='all', kind='borderline-1'):
    borderline_smote = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=42, kind=kind)
    x_resampled, y_resampled = borderline_smote.fit_resample(x, y)
    return x_resampled, y_resampled


def perform_safe_level_smote(df, target_column):
    sls = SLS()
    balanced_df = sls.balance(df, target_column)
    return balanced_df


def perform_adasyn(x, y, sampling_strategy='minority'):
    adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
    x_resampled, y_resampled = adasyn.fit_resample(x, y)
    return x_resampled, y_resampled


def perform_kmeans_smote(x, y, kmeans_estimator=None, n_clusters=5, cluster_balance_threshold=0.1):
    if kmeans_estimator is None:
        kmeans_estimator = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
    sm = KMeansSMOTE(
        kmeans_estimator=kmeans_estimator,
        random_state=42,
        cluster_balance_threshold=cluster_balance_threshold
    )
    x_resampled, y_resampled = sm.fit_resample(x, y)
    return x_resampled, y_resampled


def perform_dbscan_bsmote(x, y):
    # dbscan_bsmote = ClusterOverSampler(oversampler=BorderlineSMOTE(random_state=5), clusterer=DBSCAN())
    # x_resampled, y_resampled = dbscan_bsmote.fit_resample(x, y)
    # funziona solo con py 12
    return x, y


def perform_gaussian_mixture_clustering(x, n_components=2, random_state=0):
    gm = GaussianMixture(n_components=n_components, random_state=random_state).fit(x)
    cluster_centers = gm.means_
    labels = gm.predict(x)
    return cluster_centers, labels


def perform_oversampling_method(x, y, method_index):
    if method_index == ROS:
        x_resampled, y_resampled = perform_ROS(x, y)
    elif method_index == SMOTE_METHOD:
        x_resampled, y_resampled = perform_smote(x, y)
    elif method_index == BORDERLINE_SMOTE:
        x_resampled, y_resampled = perform_borderline_smote(x, y)
    elif method_index == ADASYN_METHOD:
        x_resampled, y_resampled = perform_adasyn(x, y)
    elif method_index == KMEANS_SMOTE:
        x_resampled, y_resampled = perform_kmeans_smote(x, y)
    elif method_index == DBSCAN_SMOTE:
        x_resampled, y_resampled = perform_dbscan_bsmote(x, y)
    else:
        return x, y

    return x_resampled, y_resampled


def create_and_save_bayesian_network(description_file='dataset_description.json',
                                     dataset_file='train_data_deep.csv',
                                     epsilon=0, degree_of_bayesian_network=2):
    if not os.path.exists(description_file):
        describer = DataDescriber()
        # Describe the dataset to create a Bayesian network
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file=dataset_file,
                                                                epsilon=epsilon,
                                                                k=degree_of_bayesian_network)
        # Save dataset description to a JSON file
        describer.save_dataset_description_to_file(description_file)
        print("Bayesian network description saved.")
        # Optionally display the Bayesian network
        # display_bayesian_network(describer.bayesian_network)
    else:
        print(f"The {description_file} file already exists. No need to run the code.")


def generate_and_save_synthetic_data_with_bayesian_network(synthetic_data_file='synthetic_data.csv',
                                                           description_file='dataset_description.json',
                                                           num_tuples_to_generate=100000):
    create_and_save_bayesian_network()

    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
    # Save synthetic data to a CSV file
    generator.save_synthetic_data(synthetic_data_file)
    print("Synthetic data generated and saved.")


def generate_and_save_synthetic_data_with_ctgan(train_data):
    # Assumptions for CTGAN training
    num_cols = list(train_data.columns[train_data.columns != 'y'])
    cat_cols = ['y']

    model_file_path = 'ctgan_model.pkl'
    if not os.path.exists(model_file_path):
        batch_size = 500
        epochs = 500
        learning_rate = 2e-4
        beta_1 = 0.5
        beta_2 = 0.9

        ctgan_args = ModelParameters(batch_size=batch_size,
                                     lr=learning_rate,
                                     betas=(beta_1, beta_2))

        train_args = TrainParameters(epochs=epochs)

        synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
        synth.fit(data=train_data, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

        synth.save(model_file_path)
        print("CTGAN model trained and saved")
    else:
        synth = RegularSynthesizer.load(model_file_path)

    synthetic_data = synth.sample(100000)
    synthetic_data.to_csv('synthetic_data.csv', index=False)
    print("Synthetic data generated and saved.")


def generate_and_save_synthetic_data_with_cgan(train_data, num_samples_per_class):
    """
    Generate synthetic data using CGAN for both classes 0 and 1.

    Args:
        train_data (DataFrame): The training data used to fit the CGAN model.
        num_samples_per_class (int): Number of synthetic samples to generate per class.

    Returns:
        DataFrame: A DataFrame containing the generated synthetic data for both classes.
    """
    # Assumptions for CGAN training
    label_cols = ["y"]  # The name of the column representing the class
    num_cols = train_data.drop(columns=label_cols, errors='ignore').columns.tolist()
    cat_cols = []  # Update this if you have categorical columns

    # CGAN Model Parameters
    gan_args = ModelParameters(batch_size=128, lr=5e-4, betas=(0.5, 0.9), noise_dim=32, layers_dim=128)
    train_args = TrainParameters(epochs=100, label_dim=1, labels=(0, 1))

    # Initialize CGAN
    synth = RegularSynthesizer(modelname='cgan', model_parameters=gan_args)
    synth.fit(data=train_data, label_cols=label_cols, train_arguments=train_args, num_cols=num_cols,
              cat_cols=cat_cols)

    synth.save('cgan_model.pkl')

    # Generate synthetic data for class 0
    cond_array_0 = pd.DataFrame(num_samples_per_class * [0], columns=label_cols)
    synthetic_data_0 = synth.sample(cond_array_0)

    # Generate synthetic data for class 1
    cond_array_1 = pd.DataFrame(num_samples_per_class * [1], columns=label_cols)
    synthetic_data_1 = synth.sample(cond_array_1)

    # Combine the synthetic data from both classes
    synthetic_data = pd.concat([synthetic_data_0, synthetic_data_1], ignore_index=True)
    synthetic_data.to_csv('synthetic_data.csv', index=False)
    print("Synthetic data generated and saved.")


def generate_and_save_synthetic_data_with_wgan(train_data):
    num_cols = list(train_data.columns[train_data.columns != 'y'])
    cat_cols = ['y']

    synthetic_data_list = []  # List to hold synthetic data from both classes

    for y_value in [0, 1]:
        # Filter the train_data for the current y_value
        filtered_train_data = train_data[train_data['y'] == y_value]

        model_file_path = f'wgan_model_y_{y_value}.pkl'  # Unique model file for each y_value
        if not os.path.exists(model_file_path):
            # WGAN training parameters
            batch_size = 128
            epochs = 500
            learning_rate = 5e-4
            beta_1 = 0.5
            beta_2 = 0.9

            wgan_args = ModelParameters(batch_size=batch_size,
                                        lr=learning_rate,
                                        betas=(beta_1, beta_2),
                                        noise_dim=32,
                                        layers_dim=128)

            train_args = TrainParameters(epochs=epochs)

            # Initialize and train a WGAN for the current y_value
            synth = RegularSynthesizer(modelname='wgan', model_parameters=wgan_args, n_critic=10)
            synth.fit(data=filtered_train_data, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

            # Save the trained model
            synth.save(model_file_path)
            print(f"WGAN model for y = {y_value} trained and saved")
        else:
            # Load the existing WGAN model for the current y_value
            synth = RegularSynthesizer.load(model_file_path)

        # Sample synthetic data using the trained model for the current y_value
        synthetic_data = synth.sample(50000)
        synthetic_data_list.append(synthetic_data)

    # Merge the synthetic data from both classes
    combined_synthetic_data = pd.concat(synthetic_data_list, ignore_index=True)

    # devo fare lo scaling perchè per qualche motivo i dati sintetici sono nuovamente non scalati
    y_column = combined_synthetic_data['y']
    data_to_scale = combined_synthetic_data.drop(columns='y')
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data_to_scale), columns=data_to_scale.columns)
    scaled_data_with_y = pd.concat([scaled_data, y_column.reset_index(drop=True)], axis=1)
    scaled_data_with_y.to_csv('synthetic_data.csv', index=False)
    print("Synthetic data generated and saved.")


def generate_and_save_synthetic_data_with_TVAE(train_data):
    model_file_path = 'TVAE_model.pkl'
    if not os.path.exists(model_file_path):
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train_data)

        synthesizer = TVAESynthesizer(metadata)
        synthesizer.fit(train_data)
        synthesizer.save(
            filepath=model_file_path
        )
        print("TVAE model trained and saved")
    else:
        synthesizer = TVAESynthesizer.load(
            filepath=model_file_path
        )

    synthetic_data = synthesizer.sample(num_rows=100000)
    synthetic_data.to_csv('synthetic_data.csv', index=False)
    print("Synthetic data generated and saved.")


def generate_and_save_synthetic_data_with_CopulaGAN(train_data):
    model_file_path = 'CopulaGAN_model.pkl'
    if not os.path.exists(model_file_path):
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train_data)

        synthesizer = CopulaGANSynthesizer(metadata)
        synthesizer.fit(train_data)
        synthesizer.save(
            filepath=model_file_path
        )
        print("CopulaGAN model trained and saved")
    else:
        synthesizer = TVAESynthesizer.load(
            filepath=model_file_path
        )

    synthetic_data = synthesizer.sample(num_rows=100000)
    synthetic_data.to_csv('synthetic_data.csv', index=False)
    print("Synthetic data generated and saved.")


def perform_oversampling_deep_method(train_data, method_index):
    if method_index == BAYESIAN_NETWORK:
        generate_and_save_synthetic_data_with_bayesian_network()
    elif method_index == CTGAN:
        generate_and_save_synthetic_data_with_ctgan(train_data)
    elif method_index == WGAN:
        generate_and_save_synthetic_data_with_wgan(train_data)
    elif method_index == TVAE:
        generate_and_save_synthetic_data_with_TVAE(train_data)
    elif method_index == COPULA_GAN:
        generate_and_save_synthetic_data_with_CopulaGAN(train_data)
    elif method_index == CGAN:
        generate_and_save_synthetic_data_with_cgan(train_data)
    else:
        raise Exception("No method found!!!")

