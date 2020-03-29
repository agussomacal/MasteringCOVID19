from typing import Dict

import numpy as np
import pandas as pd

CATEGORY_VARIABLE = 'category_variable'
INTEGRATION_VARIABLE = 'integration_variable'
MODEL_VARIABLE = 'model_variable'
COLUMNS_SPECIFICATION_TYPES = [MODEL_VARIABLE, INTEGRATION_VARIABLE, CATEGORY_VARIABLE]


class DataForModel:
    def __init__(self, data: pd.DataFrame, columns_specifications: Dict, model_vars_map2columns: Dict):
        integration_vars = [k for k, cs in columns_specifications.items() if INTEGRATION_VARIABLE == cs]
        assert len(integration_vars) == 1, 'Only 1 variable should be the {}, but found {}.'.format(INTEGRATION_VARIABLE, integration_vars)

        self.data = data
        self.columns_specifications = columns_specifications
        self.integration_variable_column_name = list(columns_specifications.keys())[
            list(columns_specifications.values()).index(INTEGRATION_VARIABLE)]
        self.model_vars_map2columns = model_vars_map2columns

    def get_data_after_category_specification(self, chosen_categories_dict):
        assert set(chosen_categories_dict.keys()) == set([k for k, cs in self.columns_specifications.items() if
                                                          cs == CATEGORY_VARIABLE]), 'All categories must have a chosen '
        data = self.data.copy()
        for category_name, chosen_category in chosen_categories_dict.items():
            data = data.loc[data[category_name] == chosen_category, ~data.columns.isin(category_name)]
        data = data.sort_values(by=self.integration_variable_column_name)
        return data

    def get_initial_condition_for_integration(self, initial_condition_dict: Dict):
        t0_ix = self.data[self.integration_variable_column_name].argmin()
        for k, v in initial_condition_dict.items():
            if v is None:
                assert k in self.model_vars_map2columns.keys(), 'there should be a representation in map'
                initial_condition_dict[k] = self.data.loc[t0_ix, self.model_vars_map2columns[k]]
        return initial_condition_dict

    def get_values_of_integration_variable(self):
        return np.sort(self.data[self.integration_variable_column_name].values)

    def get_observed_variables(self, model_var_names_as_columns=True):
        data = self.data.loc[:, self.model_vars_map2columns.values()]
        if model_var_names_as_columns:
            data.columns = [mcn for cn, mcn in zip(data.columns, self.model_vars_map2columns.keys())]
        return data