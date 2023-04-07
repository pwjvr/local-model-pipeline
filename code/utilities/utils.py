"""
General utility functions
"""
import logging
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

def setup_logging(filename):
    """
    Set up logging with basic configuration
    """
    fh = logging.FileHandler(filename)
    sh = logging.StreamHandler()
    fmt = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=fmt, handlers=[fh,sh], level=logging.INFO)

class LoggerMixin():
    """
    Solves pickling problem of loggers
    """
    @property
    def logger(self):
        component = "{}.{}".format(type(self).__module__, type(self).__name__)
        return logging.getLogger(component)

_marker = object()

def first(iterable, default=_marker):
    """
    Return the first item of *iterable*, or *default* if *iterable* is
    empty.

        >>> first([0, 1, 2, 3])
        0
        >>> first([], 'some default')
        'some default'

    If *default* is not provided and there are no items in the iterable,
    raise ``ValueError``.

    :func:`first` is useful when you have a generator of expensive-to-retrieve
    values and want any arbitrary one. It is marginally shorter than
    ``next(iter(iterable), default)``.

    """
    try:
        return next(iter(iterable))
    except StopIteration as e:
        if default is _marker:
            raise ValueError(
                'first() was called on an empty iterable, and no '
                'default value was provided.'
            ) from e
        return default
    
def columns_with_nans(df):
    mask = df.isnull().any()
    vars_with_missing = mask[mask == True].index.toList()
    return vars_with_missing

def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst
    """
    for i in range(0, len(lst), n):
        yield lst[i:i+n]