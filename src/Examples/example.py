import pandas as pd
import numpy as np

from src.DataManager import INTEGRATION_VARIABLE, MODEL_VARIABLE, DataForModel
from src.MasterFitter import MasterFitter
from src.Models.SIR_model import SIR
from src.metrics import mse

t = np.arange(20)
data = DataForModel(data=pd.DataFrame(np.asarray([np.exp(0.25*t), t]).T, columns=['Recuperados', 'Dias']),
                    columns_specifications={'Recuperados': MODEL_VARIABLE, 'Dias': INTEGRATION_VARIABLE},
                    model_vars_map2columns={'R': 'Recuperados'})

master_fitter = MasterFitter(data=data,
                             model_class=SIR,
                             initial_condition_dict={'I': 1, 'S': 1000, 'R': None},
                             metric=mse, iterations_cma=10000, sigma_cma=1, popsize=10)
coefs = master_fitter.fit_model(init_params={'b': 0.1, 'k': 0.1})
master_fitter.plot()
