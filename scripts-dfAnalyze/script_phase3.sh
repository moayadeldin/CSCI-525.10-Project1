#!/bin/bash

#####################################################################################

# This bash script is used for running df-Analyze for Phase 3 4 different times

# This code is a collaborative work between:
#
#    Moayadeldin Hussain
#    Muhammad Javed
#    Salal Ali Khan
#
# For CSCI-525.10 Project 1 Coursework submitted to Dr. Jacob Levmann.
#####################################################################################

### Target: Price, With Keeping number_of_reviews feature ###

python3 df-analyze.py \
    --df data_preprocessed_phase3.csv \
    --target price \
    --mode regress \
    --regressors knn lgbm rf elastic sgd dummy \
    --feat-select filter embed wrap \
    --redundant-wrapper-selection \
    --embed-select lgbm linear \
    --wrapper-select step-up \
    --wrapper-model linear \
    --norm robust \
    --nan median \
    --filter-method assoc pred \
    --filter-assoc-cont-regress mut_info \
    --filter-assoc-cat-regress mut_info \
    --filter-pred-regress mae \
    --htune-trials 50 \
    --htune-reg-metric mae \
    --test-val-size 0.4 \
    --outdir ./phase3_results \

### Target: Minimum Nights, With Keeping number_of_reviews feature ###

python3 df-analyze.py \
    --df data_preprocessed_phase3.csv \
    --target minimum_nights \
    --mode regress \
    --regressors knn lgbm rf elastic sgd dummy \
    --feat-select filter embed wrap \
    --redundant-wrapper-selection \
    --embed-select lgbm linear \
    --wrapper-select step-up \
    --wrapper-model linear \
    --norm robust \
    --nan median \
    --filter-method assoc pred \
    --filter-assoc-cont-regress mut_info \
    --filter-assoc-cat-regress mut_info \
    --filter-pred-regress mae \
    --htune-trials 50 \
    --htune-reg-metric mae \
    --test-val-size 0.4 \
    --outdir ./phase3_results \

### Target: Price, With Removing number_of_reviews feature ###
python3 df-analyze.py \
    --df data_preprocessed_phase3_without_number_of_reviews.csv \
    --target price \
    --mode regress \
    --regressors knn lgbm rf elastic sgd dummy \
    --feat-select filter embed wrap \
    --redundant-wrapper-selection \
    --embed-select lgbm linear \
    --wrapper-select step-up \
    --wrapper-model linear \
    --norm robust \
    --nan median \
    --filter-method assoc pred \
    --filter-assoc-cont-regress mut_info \
    --filter-assoc-cat-regress mut_info \
    --filter-pred-regress mae \
    --htune-trials 50 \
    --htune-reg-metric mae \
    --test-val-size 0.4 \
    --outdir ./phase3_results \

### Target: Minimum Nights, With Removing number_of_reviews feature ###
python3 df-analyze.py \
    --df data_preprocessed_phase3_without_number_of_reviews.csv \
    --target minimum_nights \
    --mode regress \
    --regressors knn lgbm rf elastic sgd dummy \
    --feat-select filter embed wrap \
    --redundant-wrapper-selection \
    --embed-select lgbm linear \
    --wrapper-select step-up \
    --wrapper-model linear \
    --norm robust \
    --nan median \
    --filter-method assoc pred \
    --filter-assoc-cont-regress mut_info \
    --filter-assoc-cat-regress mut_info \
    --filter-pred-regress mae \
    --htune-trials 50 \
    --htune-reg-metric mae \
    --test-val-size 0.4 \
    --outdir ./phase3_results \
