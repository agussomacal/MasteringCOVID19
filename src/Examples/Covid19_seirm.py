import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from src import config
from src.DataManager import INTEGRATION_VARIABLE, MODEL_VARIABLE, DataForModel, UNUSE_VARIABLE, CATEGORY_VARIABLE
from src.MasterFitter import MasterFitter
from src.Models.SEIRM import SEIRM
from src.Models.SIR_model import SIR
from src.config import check_create_path
from src.metrics import mse

model_class = SEIRM
extra_future_predict = 0.3

contry_pop_dict = {'Italy': 60*1e6, 'France': 67*1e6, 'Argentina': 40*1e6, 'Spain': 47*1e6}
model_vars_map2columns = {'R': 'recovered', 'I': 'confirmed', 'M': 'deaths'}
init_params = None  # {'a': 0.01, 'b':, 'c', 'd', 'm'}
initial_condition_dict = {'S': 10000, 'E': 8, 'I': 2, 'R': None, 'M': None}

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


filename_dict = {'confirmed': "time_series_covid19_confirmed_global_narrow.csv",
                 'recovered': "time_series_covid19_recovered_global_narrow.csv",
                 'deaths': "time_series_covid19_deaths_global_narrow.csv"}

data2model = pd.DataFrame([])
for k, v in filename_dict.items():
    data = pd.read_csv('{}/{}'.format(config.data_dir, v))
    data = data.groupby('Country/Region').apply(preprocess, k)
    data2model = pd.concat([data2model, data], axis=1)
data2model = data2model.reset_index()

# ----------------- run model --------------------------
for country in contry_pop_dict.keys():
    chosen_categories_dict = {'Country/Region': country}

    data = DataForModel(data=data2model,
                        columns_specifications=columns_specifications,
                        model_vars_map2columns=model_vars_map2columns)

    initial_condition_dict['S'] = contry_pop_dict[country]
    master_fitter = MasterFitter(
        data=data.get_data_after_category_specification(chosen_categories_dict=chosen_categories_dict),
        model_class=SEIRM,
        initial_condition_dict=initial_condition_dict,
        metric=mse, iterations_cma=1000000, sigma_cma=1, popsize=15, restarts=5)
    coefs = master_fitter.fit_model(init_params=init_params)

    t = data.get_values_of_integration_variable()
    dict4plot = master_fitter.get_data4plot(t=np.arange(t.min(), t.max() + (t.max()-t.min())*extra_future_predict))

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
        ax[i].set_yscale('log')
    plt.savefig('{}/{}.png'.format(check_create_path(config.results_dir, model_class.__name__), country))
    plt.close()
