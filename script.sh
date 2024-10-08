#!/bin/bash

#####################################################################################
# This code is a collaborative work between:
#
#    Moayadeldin Hussain
#    Muhammad Javed
#    Salal Ali Khan
#
# For CSCI-525.10 Project 1 Coursework submitted to Dr. Jacob Levmann.
#####################################################################################

python df-analyze.py
--df preprocessed_dataset.csv
--target price
--mode regress
--regressors knn lgbm rf elastic sgd dummy
--feat-select filter embed wrap
--redundant-wrapper-selection
--embed-select lgbm linear
--wrapper-select step-up
--wrapper-model linear
--norm robust
--nan median
--filter-method assoc pred
--filter--assoc-cont-regress mut_info
--filter-assoc-cat-regress mut_info
--filter-pred-regress mae
--htune-trials 50
--htune-reg-metric mae
--test-val-size 0.4
--outdir ./data_preprocessed
