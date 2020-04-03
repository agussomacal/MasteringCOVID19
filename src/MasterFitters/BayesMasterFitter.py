# https://machinelearningmastery.com/what-is-bayesian-optimization/
# https://thuijskens.github.io/2016/12/29/bayesian-optimisation/
# https://github.com/thuijskens/bayesian-optimization

from collections import OrderedDict
from typing import Dict, Callable

import numpy as np
import pandas as pd
from hyperopt import tpe, fmin
from hyperopt import Trials
from scipy.integrate import odeint

from src.DataManager import DataForModel
from src.MasterFitters.MasterFitter import MasterFitter, Bounds
from src.metrics import mse


class BayesMasterFitter(MasterFitter):
    def __init__(self, data: DataForModel, model_class, initial_condition_dict: Dict, iterations, params_bounds: Dict = None,
                 var_bounds: Dict = None, init_params=None, out_of_bounds_cost=np.Inf, metric=mse):
        super().__init__(data=data,
                         model_class=model_class,
                         initial_condition_dict=initial_condition_dict,
                         metric=metric,
                         init_params=init_params,
                         params_bounds=params_bounds,
                         var_bounds=var_bounds,
                         out_of_bounds_cost=out_of_bounds_cost)
        self.iterations = iterations

    def optimization_method(self, objective: Callable, x0: Dict) -> OrderedDict:
        trials = Trials()
        fitted_params = fmin(fn=objective,
                             space=x0,
                             trials=trials,
                             algo=tpe.suggest,
                             max_evals=self.iterations,
                             rstate=np.random.RandomState(42))
        return OrderedDict([(k, fitted_params[k]) for k in x0.keys()])
