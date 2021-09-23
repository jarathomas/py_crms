from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import dill
from py_crms.model0 import fit_model_0
from py_crms.model1 import fit_model_1

n_data = 2
# k=j=i=0

def fit_models(k):
    lambda_bias_model_0 = np.zeros((1, 100, 7))
    lambda_bias_model_0[:] = np.nan
    acc_model_0_prop = np.zeros((1, 100, 7))
    acc_model_0_prop[:] = np.nan
    fpr_model_0_prop = np.zeros((1, 100, 7))
    fpr_model_0_prop[:] = np.nan
    fnr_model_0_prop = np.zeros((1, 100, 7))
    fnr_model_0_prop[:] = np.nan

    lambda_bias_model_1 = np.zeros((1, 100, 7))
    lambda_bias_model_1[:] = np.nan
    acc_model_1_prop = np.zeros((1, 100, 7))
    acc_model_1_prop[:] = np.nan
    fpr_model_1_prop = np.zeros((1, 100, 7))
    fpr_model_1_prop[:] = np.nan
    fnr_model_1_prop = np.zeros((1, 100, 7))
    fnr_model_1_prop[:] = np.nan

    for i in range(100):
        for j in range(7):
            file_path = f"Compare/sim_data_i_{i+1}_k_{k+1}_j_{j+1}.csv"
            df = pd.read_csv(file_path)
            sim_data = df.filter(regex="^[^n|^lambda]", axis=1)
            lambda_true = df["lambda"].iloc[0:2]
            results0 = fit_model_0(data=sim_data,
                                   classifier="LR",
                                   true_covid_name="true_c",
                                   lambda_true=lambda_true,
                                   n_data=2)
            lambda_bias_model_0[0, i, j] = results0['bias'][1]
            acc_model_0_prop[0, i, j] = results0['acc'][1]
            fpr_model_0_prop[0, i, j] = results0['fpr'][1]
            fnr_model_0_prop[0, i, j] = results0['fnr'][1]
            results1 = fit_model_1(data=sim_data,
                                   true_covid_name="true_c",
                                   n_data=2,
                                   n_iter=500,
                                   burn_in=100,
                                   lambda_true=lambda_true,
                                   pooled=True)
            lambda_bias_model_1[0, i, j] = results1['bias'][1]
            acc_model_1_prop[0, i, j] = results1['acc'][1]
            fpr_model_1_prop[0, i, j] = results1['fpr'][1]
            fnr_model_1_prop[0, i, j] = results1['fnr'][1]

            print(f"Iteration:  Job = {k},  i = {i}, j = {j} \n")

    results = {"model0": [lambda_bias_model_0,
                          acc_model_0_prop,
                          fpr_model_0_prop,
                          fnr_model_0_prop],
               "model1": [lambda_bias_model_1,
                          acc_model_1_prop,
                          fpr_model_1_prop,
                          fnr_model_1_prop]}



    return results


results = Parallel(n_jobs=5)(delayed(fit_models)(i) for i in range(5))
dill.dump_session('./compare.pkl')
# dill.load_session('./compare.pkl')
