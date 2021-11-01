# -*- coding: utf-8 -*-

"""
py_crms.model1
~~~~~~~~~~~~~~
This module supplies the function to fit model 1.
"""
from numpy import empty, nan, nan_to_num, ix_, apply_along_axis, where, prod, quantile
from numpy.random import beta, binomial
from pandas import Series, DataFrame, concat
from PyQt5.QtWidgets import QApplication

def fit_model_1(train_data, test_data, id_column_name, covid_column_name,
                app_instance, n_iter=500, burn_in=100, pooled=True):
    """
    Generative model approach.

    :param train_data: Pandas DataFrame used to train the model.
    :param test_data: Pandas DataFrame for which the model will make predictions.
    :param id_column_name: column name in data that contains the ID
    :type id_column_name: string
    :param covid_column_name: column name in data that contains true Covid status
    :type covid_column_name: string
    :param app_instance: Instance of pycrms app.
    :param n_iter: number of iterations
    :param burn_in: number of samples to disregard
    :param pooled: whether the symptom probability is treated as the same across
    all datasets, or data-set specific
    :return: dict
    """
    new_train_data = train_data.copy()
    new_train_data['train'] = True
    new_test_data = test_data.copy()
    new_test_data[covid_column_name] = nan
    new_test_data['train'] = False
    data = concat([new_train_data, new_test_data])
    np_beta = beta
    np_binom = binomial
    np_prod = prod
    c = data[covid_column_name].copy()
    c_isna = c.isna()

    x = data.drop(columns=[id_column_name, covid_column_name, 'train'])
    q = x.shape[1]

    a_lambda = 1
    b_lambda = 5
    a_phi = b_phi = 1
    lambdas = beta(a=a_lambda, b=b_lambda, size=1)
    phi_0 = beta(a=a_phi, b=b_phi, size=q)
    phi_1 = beta(a=a_phi, b=b_phi, size=q)

    n_samples = n_iter - burn_in
    sample_lambdas = empty(shape=(n_samples, 1))
    sample_c = empty(shape=(n_samples, len(c)))
    sample_prob_c = empty(shape=(n_samples, sum(c_isna)))
    
    for i in range(n_iter):
        # sample c_i
        if sum(c_isna) > 0:
            x_array = x.to_numpy()
            na_index_array = c_isna.to_numpy()
            x_array_na = nan_to_num(x_array[na_index_array], nan=0)
            term1 = lambdas * np_prod(phi_1**x_array_na, axis=1) * \
                np_prod((1 - phi_1)**(1 - x_array_na), axis=1)
            term2 = (1 - lambdas) * np_prod(phi_0**x_array_na, axis=1) * \
                np_prod((1 - phi_0)**(1 - x_array_na), axis=1)
            prob = term1 / (term1 + term2)
            c[na_index_array] = np_binom(n=1, p=prob)
        # sample lambdas
        sample_a_lambda = a_lambda + c.sum()
        sample_b_lambda = b_lambda + (1 - c).sum()
        lambdas = np_beta(a=sample_a_lambda,
                          b=sample_b_lambda,
                          size=1)
        # store
        if i >= burn_in:
            sample_lambdas[i - burn_in] = lambdas
            sample_c[i - burn_in, ] = c
            sample_prob_c[i - burn_in, ] = prob
        # sample phi
        for l in range(q):
            c_1_x_1_sum = 0
            c_1_x_0_sum = 0
            c_0_x_0_sum = 0
            c_0_x_1_sum = 0
            c_1_x_1 = (c * x.iloc[:, l]).sum()
            c_1_x_0 = (c * (1 - x.iloc[:, l])).sum()
            c_0_x_1 = ((1 - c) * x.iloc[:, l]).sum()
            c_0_x_0 = ((1 - c) * (1 - x.iloc[:, l])).sum()
            # if not pooled, using dataset specific counts to update phi
            if not pooled:
                a0_updated = a_phi + c_0_x_1
                b0_updated = b_phi + c_0_x_0
                phi_0[l] = np_beta(a=a0_updated, b=b0_updated, size=1)
                a1_updated = a_phi + c_1_x_1
                b1_updated = b_phi + c_1_x_0
                phi_1[l] = np_beta(a=a1_updated, b=b1_updated, size=1)
            else:
                c_1_x_1_sum += c_1_x_1
                c_1_x_0_sum += c_1_x_0
                c_0_x_0_sum += c_0_x_0
                c_0_x_1_sum += c_0_x_1
            # if pooled, using all counts to updated phi
                a0_updated = a_phi + c_0_x_1_sum
                b0_updated = b_phi + c_0_x_0_sum
                phi_0[l] = np_beta(a=a0_updated, b=b0_updated, size=1)
                a1_updated = a_phi + c_1_x_1_sum
                b1_updated = b_phi + c_1_x_0_sum
                phi_1[l] = np_beta(a=a1_updated, b=b1_updated, size=1)

        progress = int(100 * i/n_iter)
        app_instance.progress_bar.setValue(progress)
        QApplication.processEvents()

    if sum(c_isna) > 0:
        na_index = ix_(c_isna)[0]
        c_unobs_pred_mat = sample_c[:, na_index]
        c_unobs_pred_prob = apply_along_axis(lambda x: sum(x == 1)/len(x), 0, c_unobs_pred_mat)
        c_unobs_pred = where(c_unobs_pred_prob > 0.5, 1, 0)
    prevalence = DataFrame(sample_lambdas, columns=['prevalence'])
    index_covid_missing = data.loc[c_isna].index
    id_covid_missing = data.loc[c_isna, 'ID']
    df_predictions = Series(c_unobs_pred,
                            name='prediction',
                            index=index_covid_missing)
    sample_prob_q025 = quantile(sample_prob_c, .025, axis=0).round(4)
    sample_prob_q975 = quantile(sample_prob_c, .975, axis=0).round(4)
    sample_prob_qtiles = DataFrame({'ci95_low': sample_prob_q025,
                                    'ci95_high': sample_prob_q975},
                                   index=index_covid_missing)
    predictions = concat([id_covid_missing, df_predictions, sample_prob_qtiles], axis=1)
    results = {'prevalence': prevalence, 'predictions': predictions}
    app_instance.progress_bar.setValue(100)
    QApplication.processEvents()
    return results
