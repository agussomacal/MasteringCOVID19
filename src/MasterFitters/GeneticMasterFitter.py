from collections import OrderedDict
from typing import Dict, Callable

import cma
import numpy as np

from src.DataManager import DataForModel
from src.MasterFitters.MasterFitter import MasterFitter


class GeneticMasterFitter(MasterFitter):
    def __init__(self, data: DataForModel, model_class, initial_condition_dict: Dict, metric, iterations_cma=1000,
                 sigma_cma=1, popsize=15, restarts=10):
        super().__init__(data, model_class, initial_condition_dict, metric)
        self.restarts = restarts
        self.sigma_cma = sigma_cma
        self.popsize = popsize
        self.iterations_cma = iterations_cma

    def optimization_method(self, objective: Callable, x0: Dict) -> OrderedDict:
        fitted_params, _ = cma.fmin2(objective_function=objective,
                                     x0=list(x0.values()),
                                     restarts=self.restarts,
                                     sigma0=self.sigma_cma,
                                     options={'ftarget': -np.Inf, 'popsize': self.popsize,
                                              'maxfevals': self.popsize * self.iterations_cma})
        return OrderedDict([(k, v) for k, v in zip(x0.keys(), fitted_params)])
