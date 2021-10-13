# -*- coding: utf-8 -*-

"""
py_crms.model1
~~~~~~~~~~~~~~
This module supplies the function to fit model 1.
"""
from numpy import empty, nan, ix_, apply_along_axis, where, prod
from numpy.random import beta, binomial
from pandas import Series
from py_crms.utils import get_bias, get_fnr, get_fpr, get_acc


def fit_model_1(data, true_covid_name=None, n_data=1, n_iter=500, burn_in=100,
                lambda_true=None, pooled=True):
    """
    Generative model approach.

    :param data: data
    :type data: Pandas DataFrame
    :param true_covid_name: column name in data that contains true Covid status
    :type true_covid_name: string
    :param n_data: number of data sets
    :param n_iter: number of iterations
    :param burn_in: number of samples to disregard
    :param pooled: whether the symptom probability is treated as the same across
    all datasets, or data-set specific
    :param lambda_true: true prevalence
    :return: dict
    """
    np_beta = beta
    np_binom = binomial
    np_prod = prod
    d = data["D"].copy()
    d_values = data["D"].unique()
    c = data["C"].copy()
    n_total = c.shape[0]
    c_isna = c.isna()
    if true_covid_name in list(data):
        c_true = data[str(true_covid_name)]
    else:
        c_true = None
    select_x = "^[^D|^C|^" + str(true_covid_name) + "]"
    x = data.filter(regex=select_x)

    c_true_dn = []
    c_dn = []
    x_dn = []
    d_group = Series([0] * n_total, dtype=int)
    for i in range(n_data):
        d_group[d == d_values[i]] = i
        d_val_index = d == d_values[i]
        if c_true is not None:
            c_true_dn.append(c_true[d_val_index])
        c_dn.append(c[d_val_index])
        x_dn.append(x[d_val_index])
    na_index_dn = []
    for i in range(n_data):
        na_index_dn.append(c_dn[i].isna())
    q = x.shape[1]

    c_pred = data["C"].copy()
    a_lambda = 1
    b_lambda = 5
    a_phi = b_phi = 1
    lambdas = beta(a=a_lambda, b=b_lambda, size=n_data)
    phi_0 = beta(a=a_phi, b=b_phi, size=n_data*q).reshape((n_data, q))
    phi_1 = beta(a=a_phi, b=b_phi, size=n_data*q).reshape((n_data, q))

    n_samples = n_iter - burn_in
    sample_lambdas = empty(shape=(n_samples, n_data))
    sample_c_dn = []
    for i in range(n_data):
        sample_c_dn.append(
            empty(shape=(n_samples, len(c_dn[i])))
        )
    for i in range(n_iter):
        for j in range(n_data):
            # sample c_i
            if sum(na_index_dn[j]) > 0:
                x_array = x_dn[j].to_numpy()
                na_index_array = na_index_dn[j].to_numpy()
                x_array_na = x_array[na_index_array]
                term1 = lambdas[j] * np_prod(phi_1[j]**x_array_na, axis=1) * \
                        np_prod((1 - phi_1[j])**(1 - x_array_na), axis=1)
                term2 = (1 - lambdas[j]) * np_prod(phi_0[j]**x_array_na, axis=1) * \
                        np_prod((1 - phi_0[j])**(1 - x_array_na), axis=1)
                prob = term1 / (term1 + term2)
                c_dn[j][na_index_array] = np_binom(n=1, p=prob)
            # sample lambdas
            sample_a_lambda = a_lambda + c_dn[j].sum()
            sample_b_lambda = b_lambda + (1 - c_dn[j]).sum()
            lambdas[j] = np_beta(a=sample_a_lambda,
                                 b=sample_b_lambda,
                                 size=1)
            # store
            if i >= burn_in:
                sample_lambdas[i - burn_in, j] = lambdas[j]
                sample_c_dn[j][i - burn_in, ] = c_dn[j]
        # sample phi
        for l in range(q):
            c_1_x_1_sum = 0
            c_1_x_0_sum = 0
            c_0_x_0_sum = 0
            c_0_x_1_sum = 0
            for j in range(n_data):
                c_1_x_1 = (c_dn[j] * x_dn[j].iloc[:, l]).sum()
                c_1_x_0 = (c_dn[j] * (1 - x_dn[j].iloc[:, l])).sum()
                c_0_x_1 = ((1 - c_dn[j]) * x_dn[j].iloc[:, l]).sum()
                c_0_x_0 = ((1 - c_dn[j]) * (1 - x_dn[j].iloc[:, l])).sum()
                # if not pooled, using dataset specific counts to update phi
                if not pooled:
                    a0_updated = a_phi + c_0_x_1
                    b0_updated = b_phi + c_0_x_0
                    phi_0[j, l] = np_beta(a=a0_updated, b=b0_updated, size=1)
                    a1_updated = a_phi + c_1_x_1
                    b1_updated = b_phi + c_1_x_0
                    phi_1[j, l] = np_beta(a=a1_updated, b=b1_updated, size=1)
                else:
                    c_1_x_1_sum += c_1_x_1
                    c_1_x_0_sum += c_1_x_0
                    c_0_x_0_sum += c_0_x_0
                    c_0_x_1_sum += c_0_x_1
            # if pooled, using all counts to updated phi
            if pooled:
                a0_updated = a_phi + c_0_x_1_sum
                b0_updated = b_phi + c_0_x_0_sum
                phi_0[0:(n_data + 1), l] = np_beta(a=a0_updated, b=b0_updated, size=1)
                a1_updated = a_phi + c_1_x_1_sum
                b1_updated = b_phi + c_1_x_0_sum
                phi_1[0:(n_data + 1), l] = np_beta(a=a1_updated, b=b1_updated, size=1)
    # bias and variance of estimated lambda
    bias = get_bias(sample_lambdas, lambda_true)
    variance = sample_lambdas.var(axis=0)

    # accuracy
    acc = [nan] * n_data
    fpr = [nan] * n_data
    fnr = [nan] * n_data

    c_unobs_pred_all = []
    for i in range(n_data):
        c_unobs_true = c_true_dn[i][na_index_dn[i]]
        if sum(na_index_dn[i]) > 0:
            na_index_i = ix_(na_index_dn[i])[0]
            c_unobs_pred_mat = sample_c_dn[i][:, na_index_i]
            c_unobs_pred_prob = apply_along_axis(lambda x: sum(x==1)/len(x), 0, c_unobs_pred_mat)
            c_unobs_pred = where(c_unobs_pred_prob > 0.5, 1, 0)
            acc[i] = get_acc(c_unobs_true, c_unobs_pred)
            fpr[i] = get_fpr(c_unobs_true, c_unobs_pred)
            fnr[i] = get_fnr(c_unobs_true, c_unobs_pred)
            c_unobs_pred_all.append(c_unobs_pred)

    results = {"sample_lambdas": sample_lambdas, "pred": c_unobs_pred_all, "bias": bias,
               "variance": variance, "acc": acc, "fpr": fpr, "fnr": fnr}
    return results
