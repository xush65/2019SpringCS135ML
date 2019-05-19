# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 21:58:51 2019

@author: xush4
"""
import numpy as np
from LeastSquaresLinearRegression import LeastSquaresLinearRegressor

from evaluate_perf_metrics import (
    calc_perf_metric__absolute_error, calc_perf_metric__squared_error,
    calc_perf_metrics_for_regressor_on_dataset)

y_tr_N = np.loadtxt('data_abalone/y_train.csv', delimiter=',', skiprows=1)
y_va_N = np.loadtxt('data_abalone/y_valid.csv', delimiter=',', skiprows=1)
y_te_N = np.loadtxt('data_abalone/y_test.csv' , delimiter=',', skiprows=1)

x_tr_NF = np.loadtxt('data_abalone/x_train.csv', delimiter=',', skiprows=1)
x_va_NF = np.loadtxt('data_abalone/x_valid.csv', delimiter=',', skiprows=1)
x_te_NF = np.loadtxt('data_abalone/x_test.csv' , delimiter=',', skiprows=1)

linear_regressor_2feats = LeastSquaresLinearRegressor()
TwoFeats=np.column_stack((x_tr_NF[:,2], x_tr_NF[:,5]))
linear_regressor_2feats.fit(TwoFeats, y_tr_N)

linear_regressor_2feats.print_weights_in_sorted_order()

#print(linear_regressor_2feats.predict(TwoFeats), y_tr_N)

calc_perf_metrics_for_regressor_on_dataset(linear_regressor_2feats, TwoFeats, y_tr_N, 'Train', ['mse','mae']);
calc_perf_metrics_for_regressor_on_dataset(linear_regressor_2feats, x_va_NF[:,[2,5]], y_va_N, 'Validation', ['mse','mae']);
calc_perf_metrics_for_regressor_on_dataset(linear_regressor_2feats, x_te_NF[:,[2,5]], y_te_N, 'Test', ['mse','mae']);