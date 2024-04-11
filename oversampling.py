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
    """
    Performs Random Over Sampling (ROS) to balance the dataset.

    Parameters
    ----------
    x : DataFrame
        Features dataset.
    y : array-like
        Target variable dataset.

    Returns
    -------
    x_resampled : DataFrame
        Resampled features dataset.
    y_resampled : array-like
        Resampled target variable dataset.
    """
    ros = RandomOverSampler(random_state=0)
    x_resampled, y_resampled = ros.fit_resample(x, y)
    return x_resampled, y_resampled


def perform_smote(x, y, sampling_strategy='minority'):
    """
    Performs SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic samples.

    Parameters
    ----------
    x : DataFrame
        Features dataset.
    y : array-like
        Target variable dataset.
    sampling_strategy : str, optional
        The sampling strategy to use (default is 'minority').

    Returns
    -------
    x_resampled : DataFrame
        Resampled features dataset.
    y_resampled : array-like
        Resampled target variable dataset.
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x, y)
    return x_resampled, y_resampled


def perform_borderline_smote(x, y, sampling_strategy='all', kind='borderline-1'):
    """
    Performs Borderline SMOTE, focusing on the samples close to the decision boundary.

    Parameters
    ----------
    x : DataFrame
        Features dataset.
    y : array-like
        Target variable dataset.
    sampling_strategy : str, optional
        The sampling strategy to use (default is 'all').
    kind : str, optional
        Specifies the kind of borderline SMOTE to perform ('borderline-1' or 'borderline-2').

    Returns
    -------
    x_resampled : DataFrame
        Resampled features dataset.
    y_resampled : array-like
        Resampled target variable dataset.
    """
    borderline_smote = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=42, kind=kind)
    x_resampled, y_resampled = borderline_smote.fit_resample(x, y)
    return x_resampled, y_resampled


def perform_safe_level_smote(df, target_column):
    """
    Performs Safe-Level SMOTE, an enhanced SMOTE variant using a safety measure.

    Parameters
    ----------
    df : DataFrame
        The complete dataset including the target column.
    target_column : str
        The name of the target column.

    Returns
    -------
    balanced_df : DataFrame
        The balanced dataset after applying Safe-Level SMOTE.
    """
    sls = SLS()
    balanced_df = sls.balance(df, target_column)
    return balanced_df


def perform_adasyn(x, y, sampling_strategy='minority'):
    """
    Performs ADASYN (Adaptive Synthetic Sampling) to create synthetic samples focusing more on the harder to learn examples.

    Parameters
    ----------
    x : DataFrame
        Features dataset.
    y : array-like
        Target variable dataset.
    sampling_strategy : str, optional
        The sampling strategy to use for oversampling (default is 'minority').

    Returns
    -------
    x_resampled : DataFrame
        Resampled features dataset.
    y_resampled : array-like
        Resampled target variable dataset.
    """
    adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
    x_resampled, y_resampled = adasyn.fit_resample(x, y)
    return x_resampled, y_resampled


def perform_kmeans_smote(x, y, kmeans_estimator=None, n_clusters=5, cluster_balance_threshold=0.1):
    """
    Performs KMeans SMOTE, which combines KMeans clustering and SMOTE to generate synthetic samples.
    This method allows for specifying the number of clusters and a balancing threshold for the clusters.

    Parameters
    ----------
    x : DataFrame
        Features dataset.
    y : array-like
        Target variable dataset.
    kmeans_estimator : object, optional
        A KMeans estimator instance. If not provided, a default MiniBatchKMeans is used with n_clusters specified.
    n_clusters : int, optional
        The number of clusters to form as well as the number of centroids to generate (default is 5).
    cluster_balance_threshold : float, optional
        The threshold for balancing clusters. Clusters below this size threshold are
         considered for SMOTE (default is 0.1).

    Returns
    -------
    x_resampled : DataFrame
        Resampled features dataset.
    y_resampled : array-like
        Resampled target variable dataset.
    """
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
    """
    Performs DBSCAN B-SMOTE, which combines DBSCAN clustering and Borderline SMOTE to generate synthetic samples.
    This method focuses on areas where minority class examples are near the decision boundaries identified
    by DBSCAN clustering.

    Note: This implementation is intended for use with specific Python versions (only works with Python 12).

    Parameters
    ----------
    x : DataFrame
        Features dataset.
    y : array-like
        Target variable dataset.

    Returns
    -------
    x : DataFrame
        Features dataset without changes (as the core implementation is commented out).
    y : array-like
        Target variable dataset without changes (as the core implementation is commented out).
    """
    # dbscan_bsmote = ClusterOverSampler(oversampler=BorderlineSMOTE(random_state=5), clusterer=DBSCAN())
    # x_resampled, y_resampled = dbscan_bsmote.fit_resample(x, y)
    # funziona solo con py 12
    return x, y


def perform_gaussian_mixture_clustering(x, n_components=2, random_state=0):
    """
    Performs clustering using a Gaussian Mixture Model (GMM).
    This method estimates the parameters of a Gaussian mixture distribution to cluster the data.

    Parameters
    ----------
    x : DataFrame
        Features dataset.
    n_components : int, optional
        The number of mixture components (default is 2).
    random_state : int, optional
        The seed used by the random number generator (default is 0).

    Returns
    -------
    cluster_centers : array-like
        The mean of each mixture component.
    labels : array-like
        Labels for each point based on the component they belong to.
    """
    gm = GaussianMixture(n_components=n_components, random_state=random_state).fit(x)
    cluster_centers = gm.means_
    labels = gm.predict(x)
    return cluster_centers, labels


def perform_oversampling_method(x, y, method_index):
    """
    Selects and performs an oversampling technique based on the specified method index.
    This method acts as a dispatcher that calls different oversampling functions depending on the method chosen.

    Parameters
    ----------
    x : DataFrame or array-like
        Features dataset.
    y : array-like
        Target variable dataset.
    method_index : str
        The index of the oversampling method to use, which corresponds to predefined constants
        representing different methods (e.g., ROS, SMOTE_METHOD, BORDERLINE_SMOTE, etc.).

    Returns
    -------
    x_resampled : DataFrame
        Resampled features dataset if an oversampling method is successfully applied.
    y_resampled : array-like
        Resampled target variable dataset if an oversampling method is successfully applied.

    Notes
    -----
    Ensure that `method_index` is correctly defined and matches one of the allowed values to avoid unexpected behavior.
    """
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
                                     epsilon=0, degree_of_bayesian_network=2, display_bayesian_network=False):
    """
    Creates a Bayesian network description based on a provided dataset and saves this description to a JSON file.
    If the description file already exists, it does not overwrite it.

    Parameters
    ----------
    description_file : str, optional
        The path to the JSON file where the dataset description will be saved (default is 'dataset_description.json').
    dataset_file : str, optional
        The path to the CSV file containing the dataset to be described (default is 'train_data_deep.csv').
    epsilon : float, optional
        The privacy budget used in differential privacy (default is 0, indicating no differential privacy).
    degree_of_bayesian_network : int, optional
        The degree of the Bayesian network, specifying the maximum number of parent nodes a node can have (default is 2).
    display_bayesian_network: bool, optional
        Display the bayesian network (default is False)

    Notes
    -----
    If the description file does not exist, the method describes the dataset in correlated attribute mode with the
    specified privacy budget and degree, then saves this description.
    If differential privacy is not needed, epsilon can be set to 0.
    An optional feature to display the Bayesian network graphically is commented out but can be enabled by the user.
    """
    if not os.path.exists(description_file):
        describer = DataDescriber()
        # Describe the dataset to create a Bayesian network
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file=dataset_file,
                                                                epsilon=epsilon,
                                                                k=degree_of_bayesian_network)
        # Save dataset description to a JSON file
        describer.save_dataset_description_to_file(description_file)
        print("Bayesian network description saved.")
        if display_bayesian_network:
            display_bayesian_network(describer.bayesian_network)
    else:
        print(f"The {description_file} file already exists. No need to run the code.")


def generate_and_save_synthetic_data_with_bayesian_network(synthetic_data_file='synthetic_data.csv',
                                                           description_file='dataset_description.json',
                                                           num_tuples_to_generate=100000):
    """
    Generates synthetic data using a Bayesian network model and saves this data to a CSV file.
    The process involves creating a Bayesian network from a dataset description and using it to generate data.

    Parameters
    ----------
    synthetic_data_file : str, optional
        The path to the CSV file where the synthetic data will be saved (default is 'synthetic_data.csv').
    description_file : str, optional
        The path to the JSON file that contains the dataset description for generating the Bayesian network
        (default is 'dataset_description.json').
    num_tuples_to_generate : int, optional
        The number of synthetic tuples to generate (default is 100,000).

    Notes
    -----
    The method starts by ensuring that a Bayesian network description exists, creating it if it doesn't.
    It then uses this network to generate the specified amount of synthetic data, which is saved to a CSV file.
    """
    print("Starting computation of Bayesian...")

    create_and_save_bayesian_network()

    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
    # Save synthetic data to a CSV file
    generator.save_synthetic_data(synthetic_data_file)
    print("Synthetic data generated and saved.")


def generate_and_save_synthetic_data_with_ctgan(train_data):
    """
    Generates synthetic data using the CTGAN model (Conditional Tabular Generative Adversarial Network)
    and saves this data to a CSV file.
    The method checks for an existing trained model and uses it; if no model exists, it trains a new one.

    Parameters
    ----------
    train_data : DataFrame
        The training dataset used to train the CTGAN model. It should include both numerical and categorical features.

    Notes
    -----
    The method defines CTGAN training parameters, including batch size, learning rate, and optimizer betas.
    It then initializes and trains a CTGAN synthesizer on the provided data,
    distinguishing between numerical and categorical columns.
    If a previously saved model exists, it is loaded to generate synthetic data; otherwise, a new model is trained.
    The generated data is then saved to 'synthetic_data.csv'.
    """
    print("Starting computation of CTGAN...")

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
    Generates synthetic data using the CGAN model (Conditional Generative Adversarial Network) based on class labels
    and saves this data to a CSV file. The method initializes and trains a CGAN model if one does not already exist,
    then uses it to generate specified amounts of synthetic data for each class label.

    Parameters
    ----------
    train_data : DataFrame
        The training dataset used to train the CGAN model. It should include the target class column and any other
        numerical features.
    num_samples_per_class : int
        The number of synthetic samples to generate for each class.

    Notes
    -----
    The CGAN model is configured with specific parameters for the batch size, learning rate, optimizer betas,
    noise dimension, and layers dimension. It distinguishes between numerical and categorical columns during training.
    Once the model is trained and saved, synthetic data for each class label (0 and 1 in this case) is generated
    using condition arrays. T
    The synthetic data from each class is then combined and saved to 'synthetic_data.csv'.
    """
    print("Starting computation of CGAN...")

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
    """
    Generates synthetic data using the WGAN model (Wasserstein Generative Adversarial Network) for each class label
    and saves this data to a CSV file after scaling.
    The method initializes and trains separate WGAN models for each class if not already existing,
    then generates synthetic data.

    Parameters
    ----------
    train_data : DataFrame
        The training dataset used to train the WGAN model, which should include the target class column 'y'
        and any other features.

    Notes
    -----
    The WGAN is trained with specific parameters for batch size, learning rate, optimizer betas, noise dimension,
    and layers dimension.
    The process involves:
    - Filtering the training data by class.
    - Training a unique WGAN model for each class value (0 and 1) if no saved model exists, otherwise loading the existing model.
    - Generating synthetic data using the trained model for each class and combining them.
    - Rescaling the combined synthetic data because the synthetic data generation might disrupt the original scale.
    - Saving the rescaled synthetic data to 'synthetic_data.csv'.
    """
    print("Starting computation of WGAN...")

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
    """
    Generates synthetic data using the TVAE model (Tabular Variational Autoencoder) and saves this data to a CSV file.
    The method initializes and trains a TVAE model if one does not already exist, then uses it to generate a
    specified amount of synthetic data.

    Parameters
    ----------
    train_data : DataFrame
        The training dataset used to train the TVAE model, which should include both numerical and categorical features.

    Notes
    -----
    The TVAE model is configured using the metadata detected from the training data. The process involves:
    - Checking if a TVAE model file already exists. If it does not, a new model is trained using the training data and
    the detected metadata, then saved.
    - If the model file exists, the model is loaded from the file.
    - Synthetic data is then generated using the TVAE model, specifying the number of rows desired.
    - The generated synthetic data is saved to 'synthetic_data.csv'.
    """
    print("Starting computation of TVAE...")

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
    """
    Generates synthetic data using the CopulaGAN model (Copula Generative Adversarial Network) and saves this
    data to a CSV file.
    This method initializes and trains a CopulaGAN model if one does not already exist, then uses it
    to generate a specified amount of synthetic data.

    Parameters
    ----------
    train_data : DataFrame
        The training dataset used to train the CopulaGAN model. This dataset should include both numerical
        and categorical features.

    Notes
    -----
    The process involves:
    - Checking if a CopulaGAN model file exists. If not, a new model is trained using the detected metadata
    from the training data and then saved.
    - If the model file exists, the CopulaGAN model is loaded from the file.
    - Synthetic data is then generated using the CopulaGAN model, specifying the number of rows desired.
    - The generated synthetic data is saved to 'synthetic_data.csv'.
    """
    print("Starting computation of CopulaGAN...")

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
    """
    Selects and executes a specified synthetic data generation method based on the provided method index.
    This method serves as a dispatcher that calls various data synthesizing functions.

    Parameters
    ----------
    train_data : DataFrame
        The training dataset used as the basis for generating synthetic data. This dataset should include both numerical
        and categorical features as required by the specific synthetic method chosen.
    method_index : str
        The index of the synthetic data generation method to use. This corresponds to predefined constants representing
        different synthetic methods, such as BAYESIAN_NETWORK, CTGAN, WGAN, TVAE, COPULA_GAN, and CGAN.

    Raises
    ------
    Exception
        If `method_index` does not correspond to any predefined method, an exception is raised to indicate no valid
        method was found.

    Notes
    -----
    This method acts as a central point for choosing among different synthetic data generation techniques.
    It directly calls one of several functions:
    - `generate_and_save_synthetic_data_with_bayesian_network`
    - `generate_and_save_synthetic_data_with_ctgan`
    - `generate_and_save_synthetic_data_with_wgan`
    - `generate_and_save_synthetic_data_with_TVAE`
    - `generate_and_save_synthetic_data_with_CopulaGAN`
    - `generate_and_save_synthetic_data_with_cgan`

    Each chosen function is responsible for generating and saving synthetic data according to its own specific
    model and parameters.
    """
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

