from sklearn.feature_selection import RFE, SelectFromModel, VarianceThreshold
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
    model = DecisionTreeClassifier()
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    rfe.fit(x, y)
    selected_features = x.columns[rfe.support_]
    return selected_features


# Random Forest Importance
def random_forest_feature_selection(x, y, importance_threshold=0.01):
    model = RandomForestClassifier()
    model.fit(x, y)
    feature_importances = model.feature_importances_
    selected_features = x.columns[feature_importances >= importance_threshold]
    return selected_features


# Extra Trees Classifier
def extra_trees_feature_selection(x, y, importance_threshold=0.01):
    model = ExtraTreesClassifier()
    model.fit(x, y)
    feature_importances = model.feature_importances_
    selected_features = x.columns[feature_importances >= importance_threshold]
    return selected_features


# Information Gain (Mutual Information)
def information_gain_feature_selection(x, y, info_threshold=0.1):
    mi_scores = mutual_info_classif(x, y)
    selected_features = x.columns[mi_scores >= info_threshold]
    return selected_features


# Variance Threshold Method
def variance_threshold_feature_selection(x, threshold=0.1):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(x)
    selected_features = x.columns[selector.get_support()]
    return selected_features


def perform_feature_selection_method(x, y, method_index):
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
