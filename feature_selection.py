# Import necessary libraries
from sklearn.feature_selection import RFE, SelectFromModel, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
import numpy as np

# Recursive Feature Elimination (RFE)
def rfe_feature_selection(X, y, n_features_to_select=10):
    model = DecisionTreeClassifier()  
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    return selected_features

# Random Forest Importance
def random_forest_feature_selection(X, y, importance_threshold=0.01):
    model = RandomForestClassifier()
    model.fit(X, y)
    feature_importances = model.feature_importances_
    selected_features = X.columns[feature_importances >= importance_threshold]
    return selected_features

# Extra Trees Classifier
def extra_trees_feature_selection(X, y, importance_threshold=0.01):
    model = ExtraTreesClassifier()
    model.fit(X, y)
    feature_importances = model.feature_importances_
    selected_features = X.columns[feature_importances >= importance_threshold]
    return selected_features

# Information Gain (Mutual Information)
def information_gain_feature_selection(X, y, info_threshold=0.1):
    mi_scores = mutual_info_classif(X, y)
    selected_features = X.columns[mi_scores >= info_threshold]
    return selected_features

# Variance Threshold Method
def variance_threshold_feature_selection(X, threshold=0.1):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    selected_features = X.columns[selector.get_support()]
    return selected_features
