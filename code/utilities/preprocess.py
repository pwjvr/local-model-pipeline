import typing
import re

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

    def _get_spec_num(self) -> typing.Dict:
        """
        Return info on how numeric vars were transformed
        """
        scaler = self.pipeline.named_transformers_['num']['scaler']
        mean, std = scaler.mean_, np.sqrt(scaler.var_)
        return {'mean': mean, 'standard_deviation': std}

    def _get_spec_cat(self) -> typing.Dict:
        """
        Create some info around how cat vars was encoded
        """
        cat_spec = []
        categorical_pipeline = self.pipeline.named_transformers_['cat']['onehot']
        mappings = {f"x{i}": v for i,v in enumerate(self.cat_vars)}

        for v in categorical_pipeline.get_feature_names():
            pseudonym, *rest = v.split('_')
            src = mappings.get(pseudonym)

            rest = self._clean_value("".join(rest))

            cat_spec.append({'source_var': src, 'sklearn_var': pseudonym, 'encoded_var': v, 'suffix': ''.join(rest)})

        return cat_spec

    def _get_spec_log(self) -> typing.Dict:
        """
        Because other transformations have this - but only log10 applied so no info returned
        """
        return {}

    @staticmethod
    def _clean_value(string):
        """
        Clean up whitespace, conform to lowercase, replace spaces with dashes and replace odd chars
        """
        string = string.strip()
        string = string.lower()
        string = re.sub(' +', '-', string)
        string = re.sub(r'[^A-Za-z0-9-\.]+','-', string)
        return string

    @property
    def feature_names(self) -> typing.List[str]:
        """
        Produce human-friendly feature names
        """
        feature_names = []
        feature_names.extend(self._get_feature_names_cat())
        feature_names.extend(self._get_feature_names_log())
        feature_names.extend(self._get_feature_names_num())
        return feature_names

    @property
    def transformation_spec(self) -> typing.Dict:
        spec = {
            'num': self._get_spec_num(),
            'cat': self._get_spec_cat(),
            'log': self._get_spec_log()
        }
        return spec