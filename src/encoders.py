from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd


class CustomCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
        self.cat_features = ['cp', 'restecg', 'thal', 'slope']

    def fit(self, X, y=None):
        for feature in self.cat_features:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder.fit(X[[feature]])
            self.encoders[feature] = encoder
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.cat_features:
            encoder = self.encoders[feature]
            encoded = encoder.transform(X[[feature]])
            feature_names = encoder.get_feature_names_out([feature])
            for i, name in enumerate(feature_names):
                X[name] = encoded[:, i]
            X.drop(feature, axis=1, inplace=True)
        return X


    
class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    def fit(self, X, y=None):
        self.scaler.fit(X[self.num_features])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.num_features] = self.scaler.transform(X[self.num_features])
        return X
