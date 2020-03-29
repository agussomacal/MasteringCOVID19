from collections import OrderedDict
from typing import Callable, Dict

import matplotlib.pylab as plt
import cma
import numpy as np
import pandas as pd
from scipy.integrate import odeint

from src.DataManager import DataForModel
from src.metrics import mse


class MasterFitter:
    def __init__(self, data: DataForModel, model_class, initial_condition_dict: Dict, metric=mse, iterations_cma=1000,
                 sigma_cma=1, popsize=15):
        self.data = data
        self.model_class = model_class
        self.initial_condition_dict = initial_condition_dict

        self.fitted_params = None

        self.metric = metric

        self.sigma_cma = sigma_cma
        self.popsize = popsize
        self.iterations_cma = iterations_cma

    def fit_model(self, init_params=None):
        if init_params is None:
            init_params = self.get_init_params()

        objective_func = self.get_objective_func()
        self.fitted_params, _ = cma.fmin2(objective_function=objective_func,
                                          x0=list(init_params.values()),
                                          sigma0=self.sigma_cma,
                                          options={'ftarget': -np.Inf, 'popsize': self.popsize,
                                                   'maxfevals': self.popsize * self.iterations_cma})

        return self.fitted_params

    def get_initial_condition2integrate(self):
        initial_condition_dict = self.data.get_initial_condition_for_integration(self.initial_condition_dict)
        eqparam_names = self.model_class.get_eqparam_names(self.model_class, without_time=True)
        return [initial_condition_dict[param_name] for param_name in eqparam_names]

    def get_objective_func(self):
        eqparam_names = self.model_class.get_eqparam_names(self.model_class, without_time=True)
        y0 = self.get_initial_condition2integrate()

        def obj_func(params):
            observed_data = self.data.get_observed_variables(model_var_names_as_columns=True)
            observed_solution = self.get_particular_solution(y0, params, eqparam_names)[observed_data.columns]
            return self.metric(observed_solution, observed_data)

        return obj_func

    def get_particular_solution(self, y0, params, eqparam_names):
        solution = odeint(self.model_class(*params).get_equation_for_odeint(), y0=y0,
                          t=self.data.get_values_of_integration_variable())
        solution = pd.DataFrame(solution, columns=eqparam_names,
                                index=self.data.data[self.data.integration_variable_column_name])
        return solution

    def predict(self):
        eqparam_names = self.model_class.get_eqparam_names(self.model_class, without_time=True)
        y0 = self.get_initial_condition2integrate()
        solution = self.get_particular_solution(y0, self.fitted_params, eqparam_names)
        return solution

    def plot(self):
        observed_data = self.data.get_observed_variables(model_var_names_as_columns=True)
        solution = self.predict()
        fig, ax = plt.subplots(nrows=1, ncols=solution.shape[1], figsize=(6 * solution.shape[1], 6))
        for i, var_name in enumerate(solution):
            ax[i].plot(solution.index, solution[var_name], label='fitted {}'.format(var_name))
            if var_name in observed_data.columns:
                ax[i].plot(observed_data.index, observed_data[var_name], '.k', label='real {}'.format(var_name))
            ax[i].legend()
            ax[i].set_xlabel(solution.index.name)
            ax[i].set_ylabel(solution.index.name)
        plt.show()

    def get_init_params(self):
        return OrderedDict([(param_name, 1) for param_name in self.model_class.get_modelparam_names(self.model_class)])
