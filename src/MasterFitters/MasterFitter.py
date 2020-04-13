from collections import OrderedDict, namedtuple
from typing import Callable, Dict, Iterable

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint

from src.DataManager import DataForModel
from src.metrics import mse

Bounds = namedtuple('Bounds', ['lower', 'upper'])

OUT_OF_BOUND_COST = 1e10


class MasterFitter:
    def __init__(self, data: DataForModel, model_class, initial_condition_dict: Dict, init_params=None,
                 metric: Callable = mse, params_bounds: Dict = None, var_bounds: Dict = None,
                 out_of_bounds_cost=OUT_OF_BOUND_COST, alpha=0.75):
        self.data = data
        self.model_class = model_class

        self.initial_condition_dict = initial_condition_dict
        self.var_bounds = var_bounds

        self.init_params = init_params
        if self.init_params is None:
            self.init_params = self.get_init_params()
        self.params_bounds = params_bounds
        self.out_of_bounds_cost = out_of_bounds_cost

        self.fitted_params = {}

        self.metric = metric
        self.alpha = alpha

    def __str__(self):
        return str(self.__class__.__name__)

    def fit_model(self):
        objective_func = self.get_objective_func()
        self.fitted_params = self.optimization_method(objective=objective_func, x0=self.init_params)
        return self.fitted_params

    def optimization_method(self, objective: Callable, x0: Dict) -> OrderedDict:
        pass

    def paramvect2dict(self, params):
        if isinstance(params, dict):
            return params
        elif isinstance(params, Iterable):
            return OrderedDict(
                [(k, v) for k, v in zip(self.model_class.get_modelparam_names(self.model_class), params)])
        else:
            raise Exception('params must be a lsit or a dict!')

    def get_objective_func(self) -> Callable:
        modelvar_names = self.model_class.get_modelvar_names(self.model_class, without_time=True)
        y0 = self.get_initial_condition2integrate()

        def obj_func(params):
            params = self.paramvect2dict(params)
            observed_data = self.data.get_observed_variables(model_var_names_as_columns=True)
            solution = self.get_particular_solution(y0, params, modelvar_names)
            observed_solution = solution[observed_data.columns]

            prediction_error = self.metric(observed_solution, observed_data)
            derivative_error = self.metric(np.diff(observed_solution), np.diff(observed_data))
            return self.alpha * prediction_error + \
                   (1 - self.alpha) * derivative_error + \
                   self.penalize_out_of_range(params, self.params_bounds) + \
                   self.penalize_out_of_range(solution.min().to_dict(), self.var_bounds) + \
                   self.penalize_out_of_range(solution.max().to_dict(), self.var_bounds)

        return obj_func

    def penalize_out_of_range(self, realization_dict, bounds_dict):
        """
        Wall+linear penalization
                ......
               /
              /
             /   <- linear with slope out_of_bounds_cost
            /
            |   <- wall out_of_bounds_cost
        ____|


        :param realization_dict:
        :param bounds_dict:
        :return:
        """
        penalty = 0
        if bounds_dict is not None:
            for param_name, param_value in realization_dict.items():
                if param_name in bounds_dict.keys():
                    vals = [param_value - bounds_dict[param_name].upper,
                            bounds_dict[param_name].lower - param_value]
                    penalty += sum([0 if val < 0 else (1 + val) * self.out_of_bounds_cost for val in vals])
            return penalty
        return penalty

    def get_initial_condition2integrate(self):
        initial_condition_dict = self.data.get_initial_condition_for_integration(self.initial_condition_dict)
        modelvar_names = self.model_class.get_modelvar_names(self.model_class, without_time=True)
        return [initial_condition_dict[param_name] for param_name in modelvar_names]

    def get_particular_solution(self, y0, params, modelvar_names, t=None):
        model = self.model_class(**self.paramvect2dict(params))
        t = self.data.get_values_of_integration_variable() if t is None else t
        solution = odeint(model.get_equation_for_odeint(), y0=y0, t=t)
        solution = pd.DataFrame(solution, columns=modelvar_names, index=t)
        return solution

    def predict(self, t=None):
        modelvar_names = self.model_class.get_modelvar_names(self.model_class, without_time=True)
        y0 = self.get_initial_condition2integrate()
        solution = self.get_particular_solution(y0, list(self.fitted_params.values()), modelvar_names, t)
        return solution

    def get_data4plot(self, t=None):
        observed_data = self.data.get_observed_variables(model_var_names_as_columns=True)
        solution = self.predict(t)
        plotting_dict = dict()
        for i, var_name in enumerate(solution):
            plotting_dict[var_name] = dict()
            plotting_dict[var_name]['prediction'] = pd.Series(solution[var_name].values, name=var_name,
                                                              index=solution.index)
            if var_name in observed_data.columns:
                plotting_dict[var_name]['real data'] = pd.Series(observed_data[var_name].values, name=var_name,
                                                                 index=observed_data.index)
        return plotting_dict

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
