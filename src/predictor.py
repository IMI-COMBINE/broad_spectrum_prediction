import pickle
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from hpo_trainer import report


def generate_confusion_matrix(fingerprint_name: str, model_name: str):
    """Generate confusion matrix for the model-fingerprint combinations."""

    assert model_name in [
        "rf",
        "xgboost",
    ], "Model name must be either 'rf' or 'xgboost'."

    test_data = pd.read_csv(f"../data/splits/{fingerprint_name}_test.csv")
    test_data.drop(columns=["cmp_id"], inplace=True)
    X_test, y_test = test_data.drop("label", axis=1), test_data["label"]

    if model_name == "rf":
        model = torch.load(f"../models/{fingerprint_name}_{model_name}.pkl")
        y_pred = model.predict(X_test)
    else:
        model = pickle.load(
            open(f"../models/{fingerprint_name}_{model_name}.pickle.dat", "rb")
        )

        # Label mapping to integers
        y_test = y_test.map(
            {
                "gram-negative": 0,
                "gram-positive": 1,
                "acid-fast": 2,
                "fungi": 3,
            }
        )

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


def get_feature_importance(fingerprint_name: str):
    """Generte feature importance plot for the model-fingerprint combinations."""
    model = torch.load(f"../models/{fingerprint_name}_rf.pkl")

    train_data = pd.read_csv(f"../data/splits/{fingerprint_name}_smote_train.csv")

    X_train, _ = train_data.drop("label", axis=1), train_data["label"]

    global_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    global_importances.sort_values(ascending=False, inplace=True)
    global_importances = global_importances[:10]  # top 10 features

    return global_importances
