# -*- coding: utf-8 -*-

"""
py_crms.model0
~~~~~~~~~~~~~~
This module supplies the function to fit model 0.
"""
from numpy import nan, where
from pandas import concat, DataFrame
import statsmodels.api as sm
import xgboost as xgb


def fit_model_0(train_data, test_data, id_column_name, covid_column_name, classifier="LR"):
    """
    Fits 'model 0' with options for logistic regression or classification and
    regression trees with adaptive boosting.
    :param train_data:
    :param test_data:
    :param covid_column_name:
    :param id_column_name:
    :param classifier:
    :return: dict
    """
    y_fit = train_data[covid_column_name]
    x_fit = train_data.drop(columns=[id_column_name, covid_column_name])
    x_predict = test_data.drop(columns=[id_column_name])
    if classifier is "LR":
        model = sm.GLM(y_fit,
                       x_fit,
                       family=sm.families.Binomial(),
                       missing='drop')
        model_out = model.fit(maxiter=25)
        predict = model_out.predict(x_predict)
    else:
        dtrain = xgb.DMatrix(x_fit,
                             label=y_fit,
                             missing=nan)
        param = {"objective": "binary:logistic",
                 "eval_metric": "logloss",
                 "subsample": 0.5,
                 "eta": 0.3}
        bst = xgb.train(param,
                        dtrain,
                        num_boost_round=50)
        dpredict = xgb.DMatrix(x_predict)
        predict = bst.predict(dpredict)
        predict_0_1 = where(predict > 0.5, 1, 0)
    df_predict = DataFrame(predict_0_1,
                           columns=['prediction'],
                           index=test_data.index)
    covid_prevalence = predict.mean()
    test_id = test_data[id_column_name]
    predictions = concat([test_id, df_predict], axis=1)

    results = {"prevalence": covid_prevalence, "predictions": predictions}
    return results
