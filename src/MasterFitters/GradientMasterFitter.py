from collections import OrderedDict
from typing import Dict, Callable

import numpy as np
from scipy.optimize import minimize

from src.DataManager import DataForModel
from src.MasterFitters.MasterFitter import MasterFitter


class GradientMasterFitter(MasterFitter):
    def __init__(self, data: DataForModel, model_class, initial_condition_dict: Dict, metric, maxiter=1000):
        super().__init__(data, model_class, initial_condition_dict, metric)
        self.maxiter = maxiter

    def optimization_method(self, objective: Callable, x0: Dict) -> OrderedDict:
        res_lsq = minimize(fun=objective, x0=np.asarray(list(x0.values())), options={'maxiter': self.maxiter})
        return OrderedDict([(k, v) for k, v in zip(x0.keys(), res_lsq.x)])
