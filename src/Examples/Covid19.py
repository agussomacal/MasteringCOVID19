import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from src import config
from src.DataManager import INTEGRATION_VARIABLE, MODEL_VARIABLE, DataForModel, UNUSE_VARIABLE, CATEGORY_VARIABLE
from src.MasterFitter import MasterFitter
from src.Models.SIR_model import SIR
from src.metrics import mse

filename = "COVID-19-geographic-disbtribution-worldwide-2020-03-29.xlsx"
data = pd.read_excel('{}/{}'.format(config.data_dir, filename))


def sort_cumsum(df):
    df['dateRep'] = pd.to_datetime(df['dateRep'])
    df = df.sort_values(by='dateRep')
    df['Time'] = np.arange(df.shape[0])
    df['deaths'] = df['deaths'].cumsum()
    df['cases'] = df['cases'].cumsum()
    return df


data = data.groupby('countriesAndTerritories').apply(sort_cumsum)

t = np.arange(20)
data = DataForModel(data=data,
                    columns_specifications={'dateRep': UNUSE_VARIABLE, 'day': UNUSE_VARIABLE, 'month': UNUSE_VARIABLE,
                                            'year': UNUSE_VARIABLE, 'cases': UNUSE_VARIABLE, 'deaths': MODEL_VARIABLE,
                                            'countriesAndTerritories': CATEGORY_VARIABLE, 'geoId': UNUSE_VARIABLE,
                                            'countryterritoryCode': UNUSE_VARIABLE, 'popData2018': UNUSE_VARIABLE,
                                            'Time': INTEGRATION_VARIABLE},
                    model_vars_map2columns={'R': 'deaths'})


chosen_categories_dict = {'countriesAndTerritories': 'China'}
master_fitter = MasterFitter(data=data.get_data_after_category_specification(chosen_categories_dict=chosen_categories_dict),
                             model_class=SIR,
                             initial_condition_dict={'I': 1, 'S': 10000, 'R': None},
                             metric=mse, iterations_cma=1000000, sigma_cma=1, popsize=15, restarts=3)
coefs = master_fitter.fit_model(init_params={'b': 0.1, 'k': 0.04})

dict4plot = master_fitter.get_data4plot()

fig, ax = plt.subplots(nrows=1, ncols=len(dict4plot), figsize=(6 * len(dict4plot), 6))
plt.suptitle('Model params: {}'.format(coefs))
for i, (var_name, d) in enumerate(dict4plot.items()):
    ax[i].plot(d['prediction'].index, d['prediction'].values, label='fitted {}'.format(var_name))
    if 'real data' in d.keys():
        ax[i].plot(d['real data'].index, d['real data'].values, '.k', label='real data for {}'.format(var_name))

    ax[i].legend()
    ax[i].set_title(var_name)
    ax[i].set_xlabel(d['prediction'].index.name)
    ax[i].set_ylabel(var_name)
plt.show()