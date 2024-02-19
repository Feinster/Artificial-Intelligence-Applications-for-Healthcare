from sklearn.feature_selection import RFE, SelectFromModel, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
import numpy as np

# 1. Recursive Feature Elimination (RFE)
def rfe_feature_selection(X, y, n_features_to_select):
    model = DecisionTreeClassifier()  # You can replace this with any other classifier
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    return selected_features

# 2. Random Forest Importance
def random_forest_feature_selection(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    feature_importances = model.feature_importances_
    selected_features = X.columns[np.argsort(feature_importances)[::-1]]
    return selected_features

# 3. Extra Trees Classifier
def extra_trees_feature_selection(X, y):
    model = ExtraTreesClassifier()
    model.fit(X, y)
    feature_importances = model.feature_importances_
    selected_features = X.columns[np.argsort(feature_importances)[::-1]]
    return selected_features

# 4. Information Gain (Mutual Information)
def information_gain_feature_selection(X, y):
    mi_scores = mutual_info_classif(X, y)
    selected_features = X.columns[np.argsort(mi_scores)[::-1]]
    return selected_features

# 5. Variance Threshold Method
def variance_threshold_feature_selection(X, threshold=0.0):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    selected_features = X.columns[selector.get_support()]
    return selected_features