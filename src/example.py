import pandas as pd
import numpy as np

from src.DataManager import INTEGRATION_VARIABLE, MODEL_VARIABLE, DataForModel
from src.MasterFitter import MasterFiter
from src.Models.SIR_model import SIR
from src.metrics import mse

t = np.arange(20)
data = DataForModel(data=pd.DataFrame(np.asarray([np.exp(t), t]).T, columns=['Recuperados', 'Dias']),
                    columns_specifications={'Recuperados': MODEL_VARIABLE, 'Dias': INTEGRATION_VARIABLE},
                    model_vars_map2columns={'R': 'Recuperados'})

master_fitter = MasterFiter(metric=mse, iterations_cma=1000, sigma_cma=1, popsize=15)
coefs = master_fitter.fit_model(data=data,
                                model_class=SIR,
                                initial_condition_dict={'I': 0, 'S': 1000, 'R': None},
                                init_params=None)
print(coefs)
