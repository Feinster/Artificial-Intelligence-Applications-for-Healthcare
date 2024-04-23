from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif

RFE_METHOD = '1'
RANDOM_FOREST = '2'
EXTRA_TREES = '3'
INFORMATION_GAIN = '4'
VARIANCE_THRESHOLD = '5'


# Recursive Feature Elimination (RFE)
def rfe_feature_selection(x, y, n_features_to_select=10):
    """
    Performs feature selection using Recursive Feature Elimination (RFE) with a Decision Tree Classifier.

    Parameters
    ----------
    x : DataFrame
        Features dataset.
    y : array-like
        Target variable dataset.
    n_features_to_select : int, optional
        The number of features to select (default is 10).

    Returns
    -------
    selected_features : Index
        The names of the selected features.
    """
    model = DecisionTreeClassifier()
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    rfe.fit(x, y)
    selected_features = x.columns[rfe.support_]
    return selected_features


# Random Forest Importance
def random_forest_feature_selection(x, y, importance_threshold=0.01):
    """
    Selects features based on their importance using a Random Forest Classifier.

    Parameters
    ----------
    x : DataFrame
        Features dataset.
    y : array-like
        Target variable dataset.
    importance_threshold : float, optional
        The threshold of feature importance above which a feature is kept (default is 0.01).

    Returns
    -------
    selected_features : Index
        The names of the selected features.
    """
    model = RandomForestClassifier()
    model.fit(x, y)
    feature_importances = model.feature_importances_
    selected_features = x.columns[feature_importances >= importance_threshold]
    return selected_features


# Extra Trees Classifier
def extra_trees_feature_selection(x, y, importance_threshold=0.01):
    """
    Selects features using an Extra Trees Classifier based on the feature importances.

    Parameters
    ----------
    x : DataFrame
        Features dataset.
    y : array-like
        Target variable dataset.
    importance_threshold : float, optional
        The threshold of feature importance that must be exceeded for a feature to be retained (default is 0.01).

    Returns
    -------
    selected_features : Index
        The names of the selected features.
    """
    model = ExtraTreesClassifier()
    model.fit(x, y)
    feature_importances = model.feature_importances_
    selected_features = x.columns[feature_importances >= importance_threshold]
    return selected_features


# Information Gain (Mutual Information)
def information_gain_feature_selection(x, y, info_threshold=0.1):
    """
    Selects features based on the mutual information (information gain) between features and the target variable.

    Parameters
    ----------
    x : DataFrame
        Features dataset.
    y : array-like
        Target variable dataset.
    info_threshold : float, optional
        The threshold of information gain above which a feature is considered important (default is 0.1).

    Returns
    -------
    selected_features : Index
        The names of the selected features.
    """
    mi_scores = mutual_info_classif(x, y)
    selected_features = x.columns[mi_scores >= info_threshold]
    return selected_features


# Variance Threshold Method
def variance_threshold_feature_selection(x, threshold=0.1):
    """
    Selects features with a variance above a specified threshold.

    Parameters
    ----------
    x : DataFrame
        Features dataset.
    threshold : float, optional
        The threshold that a feature's variance must exceed to be retained (default is 0.1).

    Returns
    -------
    selected_features : Index
        The names of the selected features.
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(x)
    selected_features = x.columns[selector.get_support()]
    return selected_features


def perform_feature_selection_method(x, y, method_index):
    """
    Acts as a dispatcher to select and execute a feature selection method based on a provided index.

    Parameters
    ----------
    x : DataFrame
        Features dataset.
    y : array-like
        Target variable dataset, required by all methods except the variance threshold method.
    method_index : str
        The index of the feature selection method to use.

    Returns
    -------
    selected_features : Index
        The names of the selected features as determined by the chosen feature selection method.

    Notes
    -----
    This method delegates the task to one of the following specific feature selection methods:
    - `rfe_feature_selection`
    - `random_forest_feature_selection`
    - `extra_trees_feature_selection`
    - `information_gain_feature_selection`
    - `variance_threshold_feature_selection`

    The appropriate method is called based on the `method_index` provided, with each method having its own specific
    parameters and threshold settings.
    """
    if method_index == RFE_METHOD:
        return rfe_feature_selection(x, y)
    elif method_index == RANDOM_FOREST:
        return random_forest_feature_selection(x, y)
    elif method_index == EXTRA_TREES:
        return extra_trees_feature_selection(x, y)
    elif method_index == INFORMATION_GAIN:
        return information_gain_feature_selection(x, y)
    elif method_index == VARIANCE_THRESHOLD:
        return variance_threshold_feature_selection(x)
    else:
        return x.columns
