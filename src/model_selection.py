"""Scripts to perform train multiple classic models."""

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pycaret.classification import setup, compare_models, pull


def train_model_pipeline(
    data: pd.DataFrame, test_data: pd.DataFrame, metric: str, exp_name: str
):
    """PyCaret pipeline for training a model on a dataset, selecting the bet and saving it to disk."""

    setup(
        data=data,
        target="label",
        test_data=test_data,
        index=False,
        train_size=0.8,  # train-validation split
        low_variance_threshold=0,
        fold_strategy="stratifiedkfold",  # CV strategy
        fold=5,  # CV strategy
        experiment_name=exp_name,
        verbose=False,
    )

    compare_models(
        sort=metric,
        round=2,  # number of decimal places for metric
        include=["nb", "lr", "lightgbm", "dt", "rf", "xgboost"],
        fold=5,
        verbose=False,
    )  # Gets the best model based on the metric for hold-out set

    eval_results = pull()
    return eval_results


def _process_results(eval_results: dict):
    """Process the results of the model selection."""
    final_eval_df = []

    for fingerprint_name, results in eval_results.items():
        for (
            model_name,
            acc,
            auc,
            recall,
            precision,
            f1,
            kappa,
            mcc,
            time,
        ) in results.values:
            if "chem_phy" in fingerprint_name:
                fingerprint_name = fingerprint_name.replace("chem_phy", "chemphy")

            final_eval_df.append(
                {
                    "fingerprint": (
                        fingerprint_name
                        if "smote" not in fingerprint_name
                        else fingerprint_name.split("_")[0]
                    ),
                    "Train type": "SMOTE" if "smote" in fingerprint_name else "Orginal",
                    "Model": model_name,
                    "kappa": kappa,
                }
            )

    final_eval_df = pd.DataFrame(final_eval_df)
    final_eval_df["fingerprint"] = final_eval_df["fingerprint"].map(
        {
            "ecfp4": "ECFP4",
            "rdkit": "RDKit",
            "maccs": "MACCS",
            "mhfp6": "MHFP6",
            "chemphys": "ChemPhys",
            "erg": "ErG",
        }
    )
    return final_eval_df


def run_model_selection(experiment_type: str, metric: str = "Kappa"):
    """Run the model selection.
    :param experiment_type: str, type of experiment. Options: ["combined", "gram-positive", "gram-negative", "fungi", "all"]
    :param metric: str, metric to optimize for. Options: ["Kappa", "AUC", "Recall", "Precision", "F1", "MCC"]
    """

    eval_results = {}

    for fingerprint_name in tqdm(
        ["ecfp4", "rdkit", "maccs", "mhfp6", "erg", "chem_phys"]
    ):
        # Splitting train data into train and validation sets
        split_dir = f"../data/splits/{experiment_type}"
        train_data = pd.read_csv(f"{split_dir}/{fingerprint_name}_train.csv")
        train_data.drop(columns=["cmp_id"], inplace=True)
        train_data, valid_data = train_test_split(
            train_data, test_size=0.2, stratify=train_data["label"], random_state=42
        )

        train_with_smote = pd.read_csv(
            f"{split_dir}/{fingerprint_name}_smote_train.csv"
        )
        smote_train_data, smote_valid_data = train_test_split(
            train_with_smote,
            test_size=0.2,
            stratify=train_with_smote["label"],
            random_state=42,
        )

        results = train_model_pipeline(
            data=train_data,
            test_data=valid_data,
            metric=metric,
            exp_name=f"{fingerprint_name}",
        )

        eval_results[f"{fingerprint_name}"] = results

        smote_results = train_model_pipeline(
            data=train_with_smote,
            test_data=smote_valid_data,
            metric=metric,
            exp_name=f"{fingerprint_name}_with_smote",
        )

        eval_results[f"{fingerprint_name}_smote"] = smote_results

    return _process_results(eval_results)
