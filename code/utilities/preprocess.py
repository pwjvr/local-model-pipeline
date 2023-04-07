import typing

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectorMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utilities.utils import LoggerMixin

class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    @staticmethod
    def log_transform(x):
        return np.log10(x+1)

    def fit(self, X, y):
        return self

    def transform(self, X):
        return self.log_transform(X)

class ZScoreSelector(BaseEstimator, SelectorMixin):
    """
    Select features that exceed a standard deviation threshold or z-scores
    """
    def __init__(self, threshold=3):
        self.threshold = threshold

    def fit(self, X, y=None):
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("Expected a dataframe or numpy array")
        self.abs_zscores_ = self.zscore(X)
        if np.all(self.abs_zscores_ <= self.threshold):
            msg = "No feature in X meets the variance threshold {0:.5f}"
            if X.shape[0] == 1:
                msg += " (X contains only one sample)"
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
        return self

    def _get_support_mask(self) -> np.array:
        return (self.abs_zscores_ > self.threshold).any(xis=0).values

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            return X[:, self._get_support_mask()]
        if isinstance(X, pd.DataFrame):
            return X.loc[:,X.columns[self._get_support_mask()]]

    @staticmethod
    def zscore(X):
        return np.abs((X-X.mean()) / X.std(ddof=0.))

class Preprocessor(BaseEstimator, TransformerMixin, LoggerMixin):
    def __init__(self, cat_vars=None, num_vars=None, log_vars=None):
        self.cat_vars = cat_vars
        self.num_vars = num_vars
        self.log_vars = log_vars

        #Initialize build process
        self._build_preprocessor()

    def fit(self, X, y):
        self.pipeline.fit(X,y)
        return self

    def _get_feature_names_cat(self) -> typing.Iterator:
        """
        Get variable names for the one-hot encoded features
        """

        if not self.cat_vars:
            return []

        mapping = {f'x{i}': v for i,v in enumerate(self.cat_vars)}
        ohe_feats = self.pipeline.named_transformers_['cat']['onehot'].get_feature_names()

        feature_names = []
        for fn in ohe_feats:
            substrings = fn.split('_')
            old_fn, *rest = substrings
            rest = self._clean_value("".join(rest))
            new_fn = mapping.get(old_fn)
            new_feature_name = f"{new_fn}_{rest}"
            feature_names.append(new_feature_name)
        return iter(feature_names)

    def _get_feature_names_log(self) -> typing.Iterator:
        if not self.log_vars:
            return []
        return iter(f"{i}_log" for i in self.log_vars)



    def _get_feature_names_num(self) -> typing.Iterator:
        if not self.num_vars:
            return []
        return iter(f"{i}_scl" for i in self.num_vars)

    def transform(self, X):
        return pd.DataFrame(self.pipeline.transform(X), columns=self.feature_names)

    def _build_preprocessor(self) -> None:
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant',fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        num_pipeline = Pipeline(steps=[('scaler', StandardScaler())])

        log_pipeline = Pipeline(steps=['logger', LogTransformer()]])

        col_trans = ColumnTransformer(
            transformers=[
            ('cat', categorical_pipeline, self.cat_vars),
            ('num', num_pipeline, self.num_vars),
            ('log', log_pipeline, self.log_vars)
            ]
        )

        self.pipeline = col_trans