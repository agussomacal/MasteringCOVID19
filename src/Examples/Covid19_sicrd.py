from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import get_cmap
from hyperopt import hp

from src import config
from src.DataManager import INTEGRATION_VARIABLE, MODEL_VARIABLE, DataForModel, CATEGORY_VARIABLE
from src.MasterFitters.BayesMasterFitter import BayesMasterFitter
from src.MasterFitters.GeneticMasterFitter import GeneticMasterFitter
from src.MasterFitters.GradientMasterFitter import GradientMasterFitter
from src.MasterFitters.MasterFitter import Bounds, OUT_OF_BOUND_COST
from src.Models.SECRD import SECRD
from src.Models.SICRD import SICRD
from src.config import check_create_path
from src.metrics import mse

country_pop_dict = OrderedDict(
    [('Italy', 60 * 1e6), ('Spain', 47 * 1e6), ('France', 67 * 1e6), ('Argentina', 40 * 1e6)])


def get_var_bounds_dict(country):
    max_pop = country_pop_dict[country]
    return {
        'S': Bounds(lower=0, upper=max_pop),
        'I': Bounds(lower=0, upper=max_pop),
        'C': Bounds(lower=0, upper=max_pop),
        'M': Bounds(lower=0, upper=max_pop),
        'RD': Bounds(lower=0, upper=max_pop),
        'RI': Bounds(lower=0, upper=max_pop)
    }


def get_data2model(min_infected, model_vars_map2columns):
    def preprocess(df, dataname):
        df = df.loc[df['Province/State'].isna(), :]
        df = df.loc[df['Value'] != 0, :]
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')

        df['Time'] = np.arange(df.shape[0])
        df = df.rename(columns={'Value': dataname})
        df[dataname] = pd.to_numeric(df[dataname])
        df = df[['Time', dataname]]
        df = df.set_index(['Time'])
        return df

    def postprocess(df):
        df = df.loc[(df['confirmed'] > min_infected).cumsum() > 0, :]
        df['Time'] = np.arange(df.shape[0])  # reset time
        df = df.drop('Country/Region', axis=1)
        return df

    filename_dict = {'confirmed': "time_series_covid19_confirmed_global_narrow.csv",
                     'recovered': "time_series_covid19_recovered_global_narrow.csv",
                     'deaths': "time_series_covid19_deaths_global_narrow.csv"}

    data2model = pd.DataFrame([])
    for k, v in filename_dict.items():
        data = pd.read_csv('{}/{}'.format(config.data_dir, v))
        data = data.groupby('Country/Region').apply(preprocess, k)
        data2model = pd.concat([data2model, data], axis=1)
    data2model = data2model.reset_index()
    data2model = data2model.groupby('Country/Region').apply(postprocess)
    data2model = data2model.reset_index()

    columns_specifications = {'confirmed': MODEL_VARIABLE,
                              'recovered': MODEL_VARIABLE,
                              'deaths': MODEL_VARIABLE,
                              'Country/Region': CATEGORY_VARIABLE,
                              'Time': INTEGRATION_VARIABLE}

    data = DataForModel(data=data2model,
                        columns_specifications=columns_specifications,
                        model_vars_map2columns=model_vars_map2columns)

    return data


def plot_results(country, dict4plot, coefs):
    def plot(ax, d, var_name, col, log=False):
        ax.plot(d['prediction'].index, d['prediction'].values, c=get_cmap('tab10')(col),
                label='fitted {}'.format(var_name))
        if 'real data' in d.keys():
            ax.plot(d['real data'].index, d['real data'].values, '.k', label='real data for {}'.format(var_name))

        ax.legend()
        ax.set_title(var_name)
        ax.set_xlabel(d['prediction'].index.name)
        ax.set_ylabel(var_name)
        if log:
            ax.set_yscale('log')

    for log in [False, True]:
        fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
        fig, ax = plt.subplots(nrows=1, ncols=len(dict4plot), figsize=(6 * len(dict4plot), 6))
        plt.suptitle('{} model params: {}'.format(country, {k: np.round(v, decimals=4) for k, v in coefs.items()}))
        for i, (var_name, d) in enumerate(dict4plot.items()):
            plot(ax[i], d, var_name, i, log)
            plot(ax2, d, var_name, i, log)
        fig.savefig(
            '{}/{}_{}.png'.format(check_create_path(config.results_dir, model_class.__name__ + extra_name), country,
                                  'log' if log else ''))
        fig2.savefig(
            '{}/{}_together_{}.png'.format(check_create_path(config.results_dir, model_class.__name__ + extra_name),
                                           country, 'log' if log else ''))
        plt.close('all')


# ----------------- run model --------------------------def run():
def run(extra_name, model_class, metric, extra_future_predict, min_infected, restarts, popsize, initial_condition_dict,
        init_params, param_bounds, bayes_iter, model_vars_map2columns):
    data = get_data2model(min_infected, model_vars_map2columns)

    results = {k: [] for k in model_class.get_modelparam_names(model_class)}
    results.update({'country': []})
    for country in country_pop_dict.keys():
        chosen_categories_dict = {'Country/Region': country}

        initial_condition_dict['S'] = country_pop_dict[country]

        master_fitter = GeneticMasterFitter(
            data=data.get_data_after_category_specification(chosen_categories_dict=chosen_categories_dict),
            model_class=model_class,
            initial_condition_dict=initial_condition_dict,
            metric=metric,
            init_params=init_params,
            params_bounds=param_bounds,
            var_bounds=get_var_bounds_dict(country),
            out_of_bounds_cost=OUT_OF_BOUND_COST,
            iterations_cma=1000000,
            sigma_cma=1,
            popsize=popsize,
            restarts=restarts
        )
        coefs = master_fitter.fit_model()

        if bayes_iter is not None:
            init_params_bayes = OrderedDict([(k, hp.normal(k, np.abs(v), np.abs(v / 5))) for k, v in coefs.items()])

            master_fitter = BayesMasterFitter(
                data=data.get_data_after_category_specification(chosen_categories_dict=chosen_categories_dict),
                model_class=model_class,
                initial_condition_dict=initial_condition_dict,
                params_bounds=param_bounds,
                init_params=init_params_bayes,
                var_bounds=get_var_bounds_dict(country),
                out_of_bounds_cost=OUT_OF_BOUND_COST,
                metric=metric,
                iterations=bayes_iter
            )
            coefs = master_fitter.fit_model()

        # master_fitter = GradientMasterFitter(
        #     data=data.get_data_after_category_specification(chosen_categories_dict=chosen_categories_dict),
        #     model_class=model_class,
        #     initial_condition_dict=initial_condition_dict,
        #     metric=metric,
        #     init_params=init_params,
        #     params_bounds=param_bounds,
        #     var_bounds=get_var_bounds_dict(country),
        #     out_of_bounds_cost=OUT_OF_BOUND_COST,
        #     maxiter=1000
        # )
        # coefs = master_fitter.fit_model()

        # ---------- results ----------
        for k, v in coefs.items():
            results[k].append(v)
        results['country'].append(country)
        pd.DataFrame.from_dict(results).to_csv(
            '{}/params.csv'.format(check_create_path(config.results_dir, model_class.__name__ + extra_name)))

        # ---------- plot ----------
        t = data.get_values_of_integration_variable()
        dict4plot = master_fitter.get_data4plot(
            t=np.arange(t.min(), t.max() + (t.max() - t.min()) * extra_future_predict))
        plot_results(country, dict4plot, coefs)


if __name__ == '__main__':
    extra_name = '_cma'

    model_class = SICRD

    # metric = lambda x, y: np.log10(mse(x, y))
    metric = mse

    extra_future_predict = 0.3
    min_infected = 50

    bayes_iter = 1000
    restarts = 5  # 7
    popsize = 15

    if model_class == SICRD:
        model_vars_map2columns = {'RD': 'recovered', 'C': 'confirmed', 'M': 'deaths'}
        initial_condition_dict = {'S': None, 'I': 5 * min_infected, 'C': min_infected, 'M': None, 'RD': None, 'RI': 0}
        init_params = OrderedDict(
            [('a', 4e-4), ('b', 1e-8), ('mu', 0.02)])
        # init_params = None
        param_bounds = {
            'a': Bounds(lower=0, upper=1.0/3),
            'b': Bounds(lower=0, upper=1),
            # 'gamma1': Bounds(lower=0, upper=1.0/20),
            # 'gamma2': Bounds(lower=0, upper=10),
            'mu': Bounds(lower=0, upper=1.0/8),
        }
    else:
        raise Exception('No model class with that name.')



    # init_params_bayes = OrderedDict(
    #     [(k, hp.normal(k, v, (param_bounds[k].upper - param_bounds[k].lower) / 2)) for k, v in init_params.items()])

    run(extra_name, model_class, metric, extra_future_predict, min_infected, restarts, popsize, initial_condition_dict,
        init_params, param_bounds, bayes_iter, model_vars_map2columns)
