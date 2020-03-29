from collections import OrderedDict
from typing import Callable, Dict

import cma
import numpy as np
import pandas as pd
from scipy.integrate import odeint

from src.DataManager import DataForModel
from src.metrics import mse


class MasterFiter:
    def __init__(self, metric=mse, iterations_cma=1000, sigma_cma=1, popsize=15):
        self.metric = metric

        self.sigma_cma = sigma_cma
        self.popsize = popsize
        self.iterations_cma = iterations_cma

    def fit_model(self, data: DataForModel, model_class, initial_condition_dict: Dict, init_params=None):
        if init_params is None:
            init_params = self.get_init_params(model_class)

        objective_func = self.get_objective_func(data, model_class, initial_condition_dict)
        coefs, _ = cma.fmin2(objective_function=objective_func,
                             x0=list(init_params.values()),
                             sigma0=self.sigma_cma,
                             options={'ftarget': -np.Inf, 'popsize': self.popsize,
                                      'maxfevals': self.popsize * self.iterations_cma})
        return coefs

    def get_objective_func(self, data: DataForModel, model_class, initial_condition_dict: Dict):
        initial_condition_dict = data.get_initial_condition_for_integration(initial_condition_dict)
        eqparam_names = model_class.get_eqparam_names(model_class, without_time=True)
        y0 = [initial_condition_dict[param_name] for param_name in eqparam_names]

        def obj_func(params):
            solution = odeint(model_class(*params).get_equation_for_odeint(), y0=y0,
                              t=data.get_values_of_integration_variable())
            observed_data = data.get_observed_variables(model_var_names_as_columns=True)
            observed_solution = pd.DataFrame(solution, columns=eqparam_names)[observed_data.columns]
            return self.metric(observed_solution, observed_data)

        return obj_func

    @staticmethod
    def get_init_params(model_class):
        return OrderedDict([(param_name, 1) for param_name in model_class.get_modelparam_names(model_class)])
