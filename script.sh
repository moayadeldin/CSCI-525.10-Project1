#!/bin/bash

python df-analyze.py \
  --df out.csv \
  --target price \
  --mode regress \
  --classifiers knn lgbm rf lr sgd dummy \
  --regressors knn lgbm rf elastic sgd dummy \
  --feat-select filter embed wrap \
  --redundant-wrapper-selection \
  --embed-select lgbm linear \
  --wrapper-select step-up \
  --wrapper-model linear \
  --norm robust \
  --nan median \
  --filter-method assoc pred \
  --filter-assoc-cont-classify mut_info \
  --filter-assoc-cat-classify mut_info \
  --filter-assoc-cont-regress mut_info \
  --filter-assoc-cat-regress mut_info \
  --filter-pred-regress mae \
  --filter-pred-classify acc \
  --htune-trials 50 \
  --htune-cls-metric acc \
  --htune-reg-metric mae \
  --test-val-size 0.4 \
  --outdir ./df_analyze_results