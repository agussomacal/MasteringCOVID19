from collections import OrderedDict
from typing import Dict, Callable

import cma
import numpy as np

from src.DataManager import DataForModel
from src.MasterFitters.MasterFitter import MasterFitter, Bounds
from src.metrics import mse


class GeneticMasterFitter(MasterFitter):
    def __init__(self, data: DataForModel, model_class, initial_condition_dict: Dict, iterations_cma=1000,
                 sigma_cma=1, popsize=15, restarts=10, init_params=None, params_bounds: Dict = None,
                 var_bounds: Dict = None, out_of_bounds_cost=np.Inf, metric=mse):
        super().__init__(data=data,
                         model_class=model_class,
                         initial_condition_dict=initial_condition_dict,
                         init_params=init_params,
                         metric=metric,
                         params_bounds=params_bounds,
                         var_bounds=var_bounds,
                         out_of_bounds_cost=out_of_bounds_cost)
        self.restarts = restarts
        self.sigma_cma = sigma_cma
        self.popsize = popsize
        self.iterations_cma = iterations_cma

    def optimization_method(self, objective: Callable, x0: Dict) -> OrderedDict:
        fitted_params, _ = cma.fmin2(objective_function=objective,
                                     x0=list(x0.values()),
                                     restarts=self.restarts,
                                     restart_from_best=True,
                                     sigma0=self.sigma_cma,
                                     options={'ftarget': -np.Inf, 'popsize': self.popsize,
                                              'maxfevals': self.popsize * self.iterations_cma})
        return OrderedDict([(k, v) for k, v in zip(x0.keys(), fitted_params)])
