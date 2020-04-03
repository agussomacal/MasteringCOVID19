from collections import OrderedDict
from typing import Dict, Callable

import numpy as np
from scipy.optimize import minimize

from src.DataManager import DataForModel
from src.MasterFitters.MasterFitter import MasterFitter, Bounds
from src.metrics import mse


class GradientMasterFitter(MasterFitter):
    def __init__(self, data: DataForModel, model_class, initial_condition_dict: Dict, maxiter,
                 params_bounds: Dict = None, var_bounds: Dict = None, init_params=None,out_of_bounds_cost=np.Inf,
                 metric=mse):
        super().__init__(data=data,
                         model_class=model_class,
                         initial_condition_dict=initial_condition_dict,
                         metric=metric,
                         init_params=init_params,
                         params_bounds=params_bounds,
                         var_bounds=var_bounds,
                         out_of_bounds_cost=out_of_bounds_cost)
        self.maxiter = maxiter

    def optimization_method(self, objective: Callable, x0: Dict) -> OrderedDict:
        res_lsq = minimize(fun=objective, x0=np.asarray(list(x0.values())), options={'maxiter': self.maxiter})
        return OrderedDict([(k, v) for k, v in zip(x0.keys(), res_lsq.x)])
