"""Hyperparameter optimization for the models."""

import json
import os
import pickle
import torch
import numpy as np
import optuna
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, cohen_kappa_score

SEED = 3544


def _generate_dirs(exp_type: str):
    """Generate directories for the experiment."""
    TRIAL_PATH = f"../experiments/{exp_type}"
    os.makedirs(TRIAL_PATH, exist_ok=True)
    return TRIAL_PATH


def save_model(model, path: str, exp_name: str):
    """Saving the model to the models folder."""
    model_path = f"{path}/{exp_name}.pkl"
    torch.save(model, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)


def save_params(params: dict, path: str, exp_name: str):
    """Saving the model parameters to the models folder."""
    with open(f"{path}/{exp_name}_params.json", "w") as f:
        json.dump(params, f, indent=4, sort_keys=True, ensure_ascii=False)


def finalize_model(model, params, exp_name, model_path: str):
    """Saving the best model and parameters to the models folder."""
    save_model(model, exp_name=exp_name, path=model_path)
    save_params(params, exp_name=exp_name, path=model_path)


def save_rf_trial(trial, model, study_name, trial_dir: str):
    """Saving the model and training parameters for each trial to the experiments folder."""
    trial_name = f"trial_{trial.number}"
    trial_path = f"{trial_dir}/{study_name}"

    os.makedirs(trial_path, exist_ok=True)

    # Save model
    model_path = f"{trial_path}/{trial_name}.pkl"
    torch.save(model, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    # Save params
    with open(f"{trial_path}/{trial_name}_params.json", "w") as f:
        json.dump(trial.params, f, indent=4, sort_keys=True, ensure_ascii=False)


def objective_rf(trial, study_name, X_train, y_train, exp_type: str):
    """Objective function for the Random Forest classifier with final eval metric as kappa."""
    # Number of trees in random forest
    n_estimators = trial.suggest_int(name="n_estimators", low=100, high=500, step=100)

    # Maximum number of levels in tree
    max_depth = trial.suggest_int(name="max_depth", low=10, high=110, step=20)

    # Minimum number of samples required to split a node
    min_samples_split = trial.suggest_int(
        name="min_samples_split", low=2, high=10, step=2
    )

    # Minimum number of samples required at each leaf node
    min_samples_leaf = trial.suggest_int(name="min_samples_leaf", low=1, high=4, step=1)

    params = {
        "n_estimators": n_estimators,
        "max_features": "sqrt",
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
    }
    model = RandomForestClassifier(random_state=SEED, **params)

    kappa_scorer = make_scorer(cohen_kappa_score)

    cv_score = cross_val_score(
        model, X_train, y_train, n_jobs=-1, cv=5, scoring=kappa_scorer
    )

    trial_path = _generate_dirs(exp_type)

    # Save model and trial
    save_rf_trial(trial=trial, model=model, study_name=study_name, trial_dir=trial_path)

    cv_mean_kappa = cv_score.mean()
    return cv_mean_kappa


def save_xgboost_trial(trial, model, study_name, trial_dir: str):
    trial_name = f"trial_{trial.number}"
    trial_path = f"{trial_dir}/{study_name}"

    os.makedirs(trial_path, exist_ok=True)
    pickle.dump(model, open(f"{trial_path}/{trial_name}.pickle.dat", "wb"))

    # Save params
    with open(f"{trial_path}/{trial_name}_params.json", "w") as f:
        json.dump(trial.params, f, indent=4, sort_keys=True, ensure_ascii=False)


def kappa_scorer(preds, dtrain):
    """Kappa scorer for the XGBoost model."""
    labels = dtrain.get_label()
    y_true = np.argmax(preds, axis=1)
    return "kappa", cohen_kappa_score(y_true, labels)


def objective_xgboost(trial, study_name, X_train, y_train, label_to_idx, exp_type: str):
    """Objective function for the XGBoost classifier with final eval metric as Kappa."""
    params = {
        "verbosity": 0,
        "objective": "multi:softmax",
        "num_class": len(label_to_idx),
        "n_estimators": 1000,
        "eta": trial.suggest_float("learning_rate", 1e-2, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.1, 0.7
        ),  # Percentage of features used per tree.
        "disable_default_eval_metric": 1,
    }

    # Training data
    y_train = y_train.map(label_to_idx)

    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Optimization of kappa scoe
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-kappa")
    xgboost_model = xgb.cv(
        params,
        dtrain,
        callbacks=[pruning_callback],
        seed=SEED,
        nfold=5,
        maximize=True,
        feval=kappa_scorer,
    )

    # Save model
    trial_path = _generate_dirs(exp_type)
    save_xgboost_trial(
        trial=trial, model=xgboost_model, study_name=study_name, trial_dir=trial_path
    )

    mean_kappa = xgboost_model["test-kappa-mean"].values[-1]  # Optimized for kappa

    return mean_kappa
