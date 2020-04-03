import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from src import config
from src.DataManager import INTEGRATION_VARIABLE, MODEL_VARIABLE, DataForModel, UNUSE_VARIABLE, CATEGORY_VARIABLE
from src.MasterFitters.MasterFitter import MasterFitter
from src.Models.SECRD import SEIRM
from src.Models.SIR_model import SIR
from src.config import check_create_path
from src.metrics import mse

hopkings_data = True

for model_class in [SEIRM]:  # SIR

    if hopkings_data:

        def preprocess(df, dataname):
            df = df.loc[df['Province/State'].isna(), :]
            df = df.loc[df['Value'] != 0, :]
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')

            df['Time'] = np.arange(df.shape[0])
            df = df.rename(columns={'Value': dataname})
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

        columns_specifications = {'confirmed': MODEL_VARIABLE,
                                  'recovered': MODEL_VARIABLE,
                                  'deaths': MODEL_VARIABLE,
                                  'Country/Region': CATEGORY_VARIABLE,
                                  'Time': INTEGRATION_VARIABLE}

        if model_class == SIR:
            print('Use SEIRM instead.')
            continue
        elif model_class == SEIRM:
            model_vars_map2columns = {'R': 'recovered', 'I': 'confirmed', 'M': 'deaths'}
            init_params = None  # {'a': 0.01, 'b':, 'c', 'd', 'm'}

    else:

        filename = "COVID-19-geographic-disbtribution-worldwide-2020-03-29.xlsx"
        data2model = pd.read_excel('{}/{}'.format(config.data_dir, filename))


        def sort_cumsum(df):
            df['dateRep'] = pd.to_datetime(df['dateRep'])
            df = df.sort_values(by='dateRep')
            df['Time'] = np.arange(df.shape[0])
            df['deaths'] = df['deaths'].cumsum()
            df['cases'] = df['cases'].cumsum()
            return df


        data2model = data2model.groupby('countriesAndTerritories').apply(sort_cumsum)

        columns_specifications = {'dateRep': UNUSE_VARIABLE, 'day': UNUSE_VARIABLE,
                                  'month': UNUSE_VARIABLE,
                                  'year': UNUSE_VARIABLE, 'cases': UNUSE_VARIABLE,
                                  'deaths': MODEL_VARIABLE,
                                  'countriesAndTerritories': CATEGORY_VARIABLE,
                                  'geoId': UNUSE_VARIABLE,
                                  'countryterritoryCode': UNUSE_VARIABLE,
                                  'popData2018': UNUSE_VARIABLE,
                                  'Time': INTEGRATION_VARIABLE}
        if model_class == SIR:
            model_vars_map2columns = {'R': 'deaths'}
            init_params = {'b': 7.9, 'k': 0.65}
        elif model_class == SEIRM:
            model_vars_map2columns = {'R': 'deaths'}

    # ----------------- run model --------------------------
    for country in ['Italy', 'France', 'Argentina', 'Spain']:
        chosen_categories_dict = {'countriesAndTerritories': country}

        if model_class == SEIRM:
            data = DataForModel(data=data2model,
                                columns_specifications=columns_specifications,
                                model_vars_map2columns=model_vars_map2columns)

            master_fitter = MasterFitter(
                data=data.get_data_after_category_specification(chosen_categories_dict=chosen_categories_dict),
                model_class=SEIRM,
                initial_condition_dict={'S': 10000, 'E': 1, 'I': None, 'R': None, 'M': None},
                metric=mse, iterations_cma=1000000, sigma_cma=1, popsize=15, restarts=3)
            coefs = master_fitter.fit_model(init_params=init_params)

        elif model_class == SIR:
            data = DataForModel(data=data,
                                columns_specifications={'dateRep': UNUSE_VARIABLE, 'day': UNUSE_VARIABLE,
                                                        'month': UNUSE_VARIABLE,
                                                        'year': UNUSE_VARIABLE, 'cases': UNUSE_VARIABLE,
                                                        'deaths': MODEL_VARIABLE,
                                                        'countriesAndTerritories': CATEGORY_VARIABLE,
                                                        'geoId': UNUSE_VARIABLE,
                                                        'countryterritoryCode': UNUSE_VARIABLE,
                                                        'popData2018': UNUSE_VARIABLE,
                                                        'Time': INTEGRATION_VARIABLE},
                                model_vars_map2columns={'R': 'deaths'})

            master_fitter = MasterFitter(
                data=data.get_data_after_category_specification(chosen_categories_dict=chosen_categories_dict),
                model_class=SIR,
                initial_condition_dict={'I': 1, 'S': 10000, 'R': None},
                metric=mse, iterations_cma=1000000, sigma_cma=1, popsize=15, restarts=3)
            coefs = master_fitter.fit_model(init_params={'b': 7.9, 'k': 0.65})

        dict4plot = master_fitter.get_data4plot()

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
        # plt.savefig('{}/{}.svg'.format(check_create_path(config.results_dir, 'SIR'), country))
        plt.savefig('{}/{}.png'.format(check_create_path(config.results_dir, 'SIR'), country))
        plt.close()
