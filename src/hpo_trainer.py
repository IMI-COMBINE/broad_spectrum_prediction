import json
import os
import pickle
import torch
import pandas as pd
import numpy as np

import optuna
import xgboost as xgb

from statistics import mean

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    cohen_kappa_score,
    make_scorer,
    roc_curve,
    auc,
)

SEED = 3544


def save_model(model, exp_name):
    """Saving the model to the models folder."""

    os.makedirs("../models", exist_ok=True)

    model_path = f"../models/{exp_name}.pkl"
    torch.save(model, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)


def save_params(params, exp_name):
    """Saving the model parameters to the models folder."""

    os.makedirs("../models", exist_ok=True)

    with open(f"../models/{exp_name}_params.json", "w") as f:
        json.dump(params, f, indent=4, sort_keys=True, ensure_ascii=False)


def finalize_model(model, params, exp_name):
    """Saving the best model and parameters to the models folder."""
    save_model(model, exp_name)
    save_params(params, exp_name)


def save_trial(trial, model, study_name):
    """Saving the model and training parameters for each trial to the experiments folder."""
    trial_name = f"trial_{trial.number}"
    trial_path = f"../experiments/{study_name}"

    os.makedirs(trial_path, exist_ok=True)

    # Save model
    model_path = f"{trial_path}/{trial_name}.pkl"
    torch.save(model, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    # Save params
    with open(f"{trial_path}/{trial_name}_params.json", "w") as f:
        json.dump(trial.params, f, indent=4, sort_keys=True, ensure_ascii=False)


def objective_rf(trial, study_name, X_train, y_train):
    """Objective function for the random forest classifier."""
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

    # Save model and trial
    save_trial(trial, model, study_name)

    mean_cv_accuracy = cv_score.mean()
    return mean_cv_accuracy


def objective_xgboost(trial, study_name, X_train, y_train, label_to_idx):
    """Objective function for the XGBoost classifier."""
    params = {
        "verbosity": 0,
        "objective": "multi:softmax",
        "num_class": 4,
        "eval_metric": "auc",
        "n_estimators": 1000,
        "eta": trial.suggest_float("learning_rate", 1e-2, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.1, 0.7
        ),  # Percentage of features used per tree.
    }

    # Training data
    y_train = y_train.map(label_to_idx)

    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Optimization of kappa scoe
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")
    xgboost_model = xgb.cv(
        params, dtrain, callbacks=[pruning_callback], seed=SEED, nfold=5
    )

    # Save model
    trial_name = f"trial_{trial.number}"
    trial_path = f"../experiments/{study_name}"

    os.makedirs(trial_path, exist_ok=True)
    pickle.dump(xgboost_model, open(f"{trial_path}/{trial_name}.pickle.dat", "wb"))

    # Save params
    with open(f"{trial_path}/{trial_name}_params.json", "w") as f:
        json.dump(trial.params, f, indent=4, sort_keys=True, ensure_ascii=False)

    mean_auc = xgboost_model["test-auc-mean"].values[-1]  # Optimized for kappa

    return mean_auc


def report(y_test, y_pred, model, exp_name: str) -> pd.DataFrame:
    """Report model performance metrics and confusion matrix on test set."""

    # For AUC, need for integer labels
    label_classes = model.classes_.tolist()
    y_test_binarize = label_binarize(y_test, classes=label_classes)
    y_pred_binarize = label_binarize(y_pred, classes=label_classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(label_classes)):
        class_name = label_classes[i]
        fpr[class_name], tpr[class_name], _ = roc_curve(
            y_test_binarize[:, i], y_pred_binarize[:, i]
        )
        roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    average_fpr = np.mean([fpr[i] for i in fpr], axis=0).tolist()
    average_tpr = np.mean([tpr[i] for i in tpr], axis=0).tolist()

    report_df = pd.DataFrame(
        {
            "accuracy": accuracy_score(y_test, y_pred),
            "cohen_kappa": cohen_kappa_score(y_test, y_pred),
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
            "roc_auc": mean([roc_auc[i] for i in roc_auc]),
        },
        index=[exp_name],
    )
    return report_df, average_fpr, average_tpr
