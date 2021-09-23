# -*- coding: utf-8 -*-

"""
py_crms.utils
~~~~~~~~~~~~~~
This module provides utility functions that are used within py_crms
that are also useful for external consumption.
"""

#from math import exp
from numpy import nan

def get_bias(estimate, truth):
    """Calculate the estimated bias."""
    bias = [x - y for x, y in zip(estimate, truth)]
    return bias

def get_acc(unobs_true, unobs_pred):
    """Accuracy of individual-level classification"""
    n_unobs = len(unobs_true)
    if n_unobs == 0:
        return nan
    else:
        n_pred_correct = n_unobs - sum(abs(unobs_pred - unobs_true))
        acc = n_pred_correct / n_unobs
        return acc

def get_fpr(unobs_true, unobs_pred):
    """False positive rate of individual-level classification"""
    n_unobs = len(unobs_true)
    if n_unobs == 0:
        return nan
    else:
        fpr = sum((unobs_true == 0) & (unobs_pred == 1)) / sum(unobs_true == 0)
        return fpr

def get_fnr(unobs_true, unobs_pred):
    """False negative rate of individual-level classification"""
    n_unobs = len(unobs_true)
    if n_unobs == 0:
        return nan
    else:
        fnr = sum((unobs_true == 1) & (unobs_pred == 0)) / sum(unobs_true == 1)
        return fnr

# def expit(x):
#     """Expit function (inverse of logit transformation)."""
#     y = exp(x)/(1 + exp(x))
#     return y


