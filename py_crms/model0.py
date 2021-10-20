# -*- coding: utf-8 -*-

"""
py_crms.model0
~~~~~~~~~~~~~~
This module supplies the function to fit model 0.
"""
from numpy import nan, where
from pandas import get_dummies
import statsmodels.api as sm
import xgboost as xgb
from py_crms.utils import get_bias, get_fnr, get_fpr, get_acc


def fit_model_0(data, classifier="LR", true_covid_name=None, lambda_true=None, n_data=1):
    """
    Fits 'model 0' with options for logistic regression or classification and
    regression trees with adaptive boosting.
    :param data:
    :param classifier:
    :param true_covid_name:
    :param lambda_true:
    :param n_data:
    :return: dict
    """
    d = data["D"].copy()
    if true_covid_name in list(data):
        c_true = data[str(true_covid_name)]
    else:
        c_true = None
    select_x = "^[^D|^" + str(true_covid_name) + "]"
    sample = data.filter(regex=select_x)
    if d.nunique() > 1:
        d_mat = get_dummies(d, prefix="d")
        sample = sample.join(d_mat)
    na_index = data["C"].isna()
    d_values = data["D"].unique()
    na_index_dn = []
    for i in range(d.nunique()):
        data_na_index_dn = data[na_index & (data["D"] == d_values[i])]
        na_index_dn.append(data_na_index_dn.index)
    sample_obs = sample[na_index.eq(False)].copy()
    y_fit = sample_obs["C"]
    x_fit = sample_obs.filter(regex="[^C]")
    x_predict = sample.filter(regex="[^C]")

    if classifier is "LR":
        model = sm.GLM(y_fit, x_fit, family=sm.families.Binomial())
        results = model.fit(maxiter=25)
        predict_full = results.predict(x_predict)
    else:
        dtrain = xgb.DMatrix(x_fit, label=y_fit, missing=nan)
        param = {"objective": "binary:logistic",
                 "eval_metric": "logloss",
                 "subsample": 0.5,
                 "eta": 0.3}
        bst = xgb.train(param, dtrain, num_boost_round=50)
        dpredict = xgb.DMatrix(x_predict)
        predict_full = bst.predict(dpredict)

    predict_unobs_dn = []
    c_unobs_pred_dn = []
    lambda_hat = []
    for i in range(n_data):
        predict_unobs_dn.append(predict_full[na_index_dn[i]])
        c_unobs_pred_dn.append(where(predict_unobs_dn[i] > 0.5, 1, 0))
        lambda_hat.append(predict_full[d == d_values[i]].mean())

    if lambda_true is None:
        bias = [nan] * n_data
    else:
        bias = get_bias(lambda_hat, lambda_true)

    variance = nan

    c_unobs_true_dn = []
    acc = [nan] * n_data
    fpr = [nan] * n_data
    fnr = [nan] * n_data
    if c_true is not None:
        for i in range(n_data):
            c_unobs_true_dn.append(c_true[na_index & (data["D"] == d_values[i])])
            acc[i] = get_acc(c_unobs_true_dn[i], c_unobs_pred_dn[i])
            fpr[i] = get_fpr(c_unobs_true_dn[i], c_unobs_pred_dn[i])
            fnr[i] = get_fnr(c_unobs_true_dn[i], c_unobs_pred_dn[i])

    results = {"lambda": lambda_hat, "pred": c_unobs_pred_dn, "bias": bias,
               "variance": variance, "acc": acc, "fpr": fpr, "fnr": fnr}
    return results
