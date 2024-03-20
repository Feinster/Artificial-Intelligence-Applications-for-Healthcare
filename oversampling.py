from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE
from crucio import SLS
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from clover.over_sampling import ClusterOverSampler
from sklearn.mixture import GaussianMixture

ROS = '1'
SMOTE_METHOD = '2'
BORDERLINE_SMOTE = '3'
ADASYN_METHOD = '5'
KMEANS_SMOTE = '6'
DBSCAN_SMOTE = '7'


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
    dbscan_bsmote = ClusterOverSampler(oversampler=BorderlineSMOTE(random_state=5), clusterer=DBSCAN())
    x_resampled, y_resampled = dbscan_bsmote.fit_resample(x, y)
    return x_resampled, y_resampled


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
