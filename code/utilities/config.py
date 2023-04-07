import os
import yaml
from pathlib import Path
import numpy as np
from utils import first

class Config:
    """
    Streamline preprocessing and data cleaning steps by specifying in YAML file and ingesting through this class
    """

    def __init__(self, path):
        self.path = path

    @property
    def data(self):
        """
        Read YAML file
        """
        with open(self.path, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        return data
    
    @property
    def metadata_variables(self, lower=True):
        """
        Gather variable names that has the metadata flag
        """
        vars = []
        for v in self.data.get('variables'):
            name = v.get('original_name')
            is_meta = v.get('metadata')
            if is_meta:
                if lower:
                    name = name.lower()
                vars.append(name)
        return vars
    
    @property
    def all_variables(self):
        """
        All vars that are metadata or has a preprocessing pipeline
        """
        vars = []
        for v in self.data.get('variables'):
            name = v.get('original_name')
            pipelines = v.get('preprocessor_pipelines', [])
            metadata = v.get('metadata', False)
            if metadata is True or (metadata is False and pipelines):
                vars.append(name)
        return vars
    
    @property
    def categorical_variables(self):
        return self._get_variables_pipelines('cat')
    
    @property
    def num_variables(self):
        return self._get_variables_pipelines('num')
    
    @property
    def log_variables(self):
        return self._get_variables_pipelines('log')
    
    def get_replacement_rules(self, lowercase=True) -> dict:
        """
        Construct a dictionary that for each var, has a mapping of how to replace the old value with a new one
        """
        replace_rules = {}
        for v in self.data.get('variables'):
            name = v.get('original_name')
            repl = v.get('replacements')
            if not repl or not name:
                continue
            rules = {}
            for r in repl:
                try:
                    old, new = list(r.keys())[0], list(r.values())[0]
                except AttributeError:
                    raise AttributeError(f'Unexpected rule. Expected `old: new`, got `{r}`')
                if old == 'np.NaN' or old == 'np.nan' or old == 'nan':
                    old = np.NaN
                if lowercase:
                    name = name.lower()
                rules[old] = new

            replace_rules[name] = rules
        return replace_rules
    
    def get_dtypes(self) -> dict:
        """
        Get the schema for the variables
        """
        dtypes = {}
        for v in self.data.get('variables'):
            name = v.get('original_name')
            dtype = v.get('original_dtype')
            if dtype not in {"category","object","bool_","int_","intc","intp","int8","int16",
                             "int32","int64","uint8","uint16","uint32","uint64","float_",
                             "float16","float32","float64","complex_","complex64","complex128"}:
                raise ValueError(f"Not a valid data type. {dtype}. Must be a valid numpy data type or object")
            if not dtype or not name:
                continue
            dtypes[name] = dtype

        return dtypes
    
    def get_consts(self):
        """
        Gather constant values set as parameters in YAML file
        """
        consts = first(self.data.get('constants'))
        if not consts:
            raise ValueError("Expected a section for `constants`")
        return consts.get('SEED'), consts.get('VALIDATION_SIZE')
    
    def _get_variables_pipelines(self, pipeline_name):
        """
        Get variable names for preprocessing
        """
        vars = []
        for v in self.data.get('variables'):
            name = v.get('original_name')
            pipelines = v.get('preprocessor_pipelines')
            if not pipelines or not name:
                continue
            if pipeline_name in pipelines:
                vars.append(name.lower())
            if pipeline_name not in pipelines:
                continue
        return vars

