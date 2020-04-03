from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from hyperopt import hp

from src import config
from src.DataManager import INTEGRATION_VARIABLE, MODEL_VARIABLE, DataForModel, CATEGORY_VARIABLE
from src.MasterFitters.BayesMasterFitter import BayesMasterFitter
from src.MasterFitters.GeneticMasterFitter import GeneticMasterFitter
from src.MasterFitters.GradientMasterFitter import GradientMasterFitter
from src.Models.SECRD import SEIRM
from src.Models.SIRM import SIRM
from src.config import check_create_path
from src.metrics import mse

extra_name = '_bayes'
MFitter = GradientMasterFitter  # GeneticMasterFitter
model_class = SEIRM
extra_future_predict = 0.3
min_infected = 50

contry_pop_dict = OrderedDict([('Italy', 60 * 1e6), ('Spain', 47 * 1e6), ('France', 67 * 1e6), ('Argentina', 40 * 1e6)])
model_vars_map2columns = {'R': 'recovered', 'I': 'confirmed', 'M': 'deaths'}
init_params = OrderedDict([('a', 0.001), ('b', 0.001), ('c', 0.001), ('d', 0.04), ('m', 0.02)])
init_params_bayes = OrderedDict([
    ('b', hp.uniform('b', 0.001, 2)),
    ('d', hp.uniform('d', 0.001, 0.1)),
    ('m', hp.uniform('m', 0.001, 0.1))
])
bayes_iter = 1000
restarts = 1
popsize = 10
initial_condition_dict = {'S': None, 'I': None, 'R': None, 'M': None}

columns_specifications = {'confirmed': MODEL_VARIABLE,
                          'recovered': MODEL_VARIABLE,
                          'deaths': MODEL_VARIABLE,
                          'Country/Region': CATEGORY_VARIABLE,
                          'Time': INTEGRATION_VARIABLE}


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

# ----------------- run model --------------------------
for country in contry_pop_dict.keys():
    chosen_categories_dict = {'Country/Region': country}

    data = DataForModel(data=data2model,
                        columns_specifications=columns_specifications,
                        model_vars_map2columns=model_vars_map2columns)

    initial_condition_dict['S'] = contry_pop_dict[country]

    master_fitter = GeneticMasterFitter(
        data=data.get_data_after_category_specification(chosen_categories_dict=chosen_categories_dict),
        model_class=model_class,
        initial_condition_dict=initial_condition_dict,
        metric=mse,
        iterations_cma=1000000,
        sigma_cma=1,
        popsize=popsize,
        restarts=restarts
    )
    coefs = master_fitter.fit_model(init_params=init_params)
    init_params_bayes = OrderedDict([(k, hp.normal(k, np.abs(v), np.abs(v / 5))) for k, v in coefs.items()])

    master_fitter = BayesMasterFitter(
        data=data.get_data_after_category_specification(chosen_categories_dict=chosen_categories_dict),
        model_class=model_class,
        initial_condition_dict=initial_condition_dict,
        metric=mse,
        iterations=bayes_iter
    )
    coefs = master_fitter.fit_model(init_params=init_params_bayes)

    # init_params = coefs#{'b': 0.001, 'd': 0.04, 'm': 0.02}
    #
    # master_fitter = MFitter(
    #     data=data.get_data_after_category_specification(chosen_categories_dict=chosen_categories_dict),
    #     model_class=model_class,
    #     initial_condition_dict=initial_condition_dict,
    #     metric=mse,
    #     maxiter=1000
    # )
    # coefs = master_fitter.fit_model(init_params=init_params)

    t = data.get_values_of_integration_variable()
    dict4plot = master_fitter.get_data4plot(t=np.arange(t.min(), t.max() + (t.max() - t.min()) * extra_future_predict))

    fig, ax = plt.subplots(nrows=1, ncols=len(dict4plot), figsize=(6 * len(dict4plot), 6))
    plt.suptitle('{} model params: {}'.format(country, {k: np.round(v, decimals=4) for k, v in coefs.items()}))
    for i, (var_name, d) in enumerate(dict4plot.items()):
        ax[i].plot(d['prediction'].index, d['prediction'].values, label='fitted {}'.format(var_name))
        if 'real data' in d.keys():
            ax[i].plot(d['real data'].index, d['real data'].values, '.k', label='real data for {}'.format(var_name))

        ax[i].legend()
        ax[i].set_title(var_name)
        ax[i].set_xlabel(d['prediction'].index.name)
        ax[i].set_ylabel(var_name)
        # ax[i].set_yscale('log')
    plt.savefig('{}/{}.png'.format(check_create_path(config.results_dir, model_class.__name__ + extra_name), country))
    plt.close()
