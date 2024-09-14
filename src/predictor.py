"""Predictor module for model evaluation and performance metrics."""

import pickle
import pandas as pd
import torch

import numpy as np
from statistics import mean

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    cohen_kappa_score,
    roc_curve,
    auc,
    confusion_matrix,
)


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


def generate_confusion_matrix(
    fingerprint_name: str, model_name: str, exp_type: str, label_classes: dict
):
    """Generate confusion matrix for the model-fingerprint combinations."""

    assert model_name in [
        "rf",
        "xgboost",
    ], "Model name must be either 'rf' or 'xgboost'."

    test_data = pd.read_csv(f"../data/splits/{exp_type}/{fingerprint_name}_test.csv")
    test_data.drop(columns=["cmp_id"], inplace=True)
    X_test, y_test = test_data.drop("label", axis=1), test_data["label"]

    if model_name == "rf":
        model = torch.load(f"../models/{exp_type}/{fingerprint_name}_{model_name}.pkl")
        y_pred = model.predict(X_test)
    else:
        model = pickle.load(
            open(
                f"../models/{exp_type}/{fingerprint_name}_{model_name}.pickle.dat", "rb"
            )
        )

        # Label mapping to integers
        y_test = y_test.map(label_classes)

        # Testing data
        y_pred = model.predict(X_test)

    # Generating the report
    report_df, fpr, tpr = report(
        y_test=y_test,
        y_pred=y_pred,
        model=model,
        exp_name=f"{fingerprint_name}_{model_name}",
    )

    label_classes = model.classes_.tolist()

    # plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=label_classes)

    return report_df, cm, fpr, tpr, label_classes
