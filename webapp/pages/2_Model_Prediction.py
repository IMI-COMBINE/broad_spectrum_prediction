# Description: This file contains the code for model prediction.

import logging
import pandas as pd
import streamlit as st
import torch
import pickle

from rdkit.Chem import (
    CanonSmiles,
    MolFromSmiles,
    rdFingerprintGenerator,
    MACCSkeys,
    Descriptors,
    rdMolDescriptors,
    rdReducedGraphs,
    GraphDescriptors,
)
from mhfp.encoder import MHFPEncoder
import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

st.set_page_config(
    layout="wide",
    page_title="AMR-KG",
    page_icon=":microscope:",
    initial_sidebar_state="auto",
)

st.markdown(
    "<h1 style='text-align: center; color: #78bc1e;'>Chemical - Strain activity prediction</h1>",
    unsafe_allow_html=True,
)

st.header(
    "🔍 Compound broad spectrum activity prediction",
    divider="orange",
    help="Prediction of chemical strain activity by best model.",
)

# Data input
cols = st.columns(2)

with cols[0]:
    uploaded_file = st.file_uploader(
        "Upload a file containing SMILES",
        help="Upload a file with one SMILES per row or a comma-separated list (CSV or TSV)",
        type=["csv"],
    )
with cols[1]:
    text_input = st.text_area(
        "Or enter SMILES (one per line)",
        "",
        help="Paste yours SMILES in each line",
    )


# Model selection
st.markdown("### Select the model to use for prediction:")

cols = st.columns(2)
with cols[0]:
    model = st.radio(
        "#### Model",
        ("Random Forest", "XGBoost"),
        index=0,
        help="Select the model to use for prediction",
        horizontal=True,
    )

    st.write("#### Select the fingerprint:")
    fingerprint = st.radio(
        "Fingerprint",
        ("MHFP6", "ECFP4", "RDKIT", "MACCS", "ErG", "ChemPhys"),
        index=0,
        help="Select the fingerprint representation to use for prediction",
    )

    st.write("**The best model combination is Random Forest with MHFP6 fingerprint.**")

with cols[1]:

    metric_df = pd.read_csv("../data/test_metrics.tsv", sep="\t").reset_index(drop=True)
    metric_df.rename(columns={"Unnamed: 0": "model_name"}, inplace=True)
    metric_df["model"] = metric_df["model_name"].apply(
        lambda x: (
            x.split("_")[1].upper()
            if len(x.split("_")) < 3
            else x.split("_")[-1].upper()
        )
    )
    metric_df["fingerprints"] = metric_df["model_name"].apply(
        lambda x: (
            x.split("_")[0].upper()
            if len(x.split("_")) < 3
            else x.split("_")[0].upper() + "_" + x.split("_")[1].upper()
        )
    )
    metric_df["fingerprints"] = metric_df["fingerprints"].replace(
        {
            "ERG": "ErG",
            "CHEM_PHYS": "ChemPhys",
        }
    )

    colors = {
        "MHFP6": "#3a2c20",
        "ECFP4": "#b65c11",
        "RDKIT": "#e7a504",
        "MACCS": "#719842",
        "ErG": "#3d8ebf",
        "ChemPhys": "#901b1b",
        "RF": "#3a2c20",
        "XGBOOST": "#719842",
    }

    metric_df["accuracy"] = metric_df["accuracy"] * 100
    plt.figure(figsize=(5, 5))
    sns.violinplot(x="model", y="accuracy", data=metric_df, palette=colors)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Model", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    st.pyplot(plt)

if st.button("Predict"):
    if uploaded_file is not None:
        smiles_df = pd.read_csv(uploaded_file, header=None)
    else:
        smiles_df = pd.DataFrame(text_input.split("\n"), columns=["smiles"])

    if model == "Random Forest":
        model_name = "rf"
    else:
        model_name = "xgboost"

    if fingerprint == "MHFP6":
        fingerprint_name = "mhfp6"
    elif fingerprint == "ECFP4":
        fingerprint_name = "ecfp4"
    elif fingerprint == "RDKIT":
        fingerprint_name = "rdkit"
    elif fingerprint == "MACCS":
        fingerprint_name = "maccs"
    elif fingerprint == "ErG":
        fingerprint_name = "erg"
    else:
        model = "chem_phys"

    logger.info("⏳ Loading models")

    if model_name == "rf":
        model = torch.load(f"../models/{fingerprint_name}_{model_name}.pkl")
    else:
        model = pickle.load(
            open(f"../models/{fingerprint_name}_{model_name}.pickle.dat", "rb")
        )

    logger.info("🔮 Processing SMILES to fingerprints")

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=4, fpSize=1024
    )  # ECFP4 fingerprint
    rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(
        fpSize=1024
    )  # RDKit fingerprint
    mhfp_encoder = MHFPEncoder(n_permutations=2048, seed=42)  # MHFP6 fingerprint

    ecfp_fingerprints = []
    rdkit_fingerprints = []
    maccs_fingerprints = []
    mhfp6_fingerprints = []
    erg_fingerprints = []
    chem_phys_props = []

    for smiles in smiles_df["smiles"].values:
        # Canonicalize the smiles
        try:
            can_smiles = CanonSmiles(smiles)
        except Exception as e:
            can_smiles = smiles

        # Generate the mol object
        mol = MolFromSmiles(can_smiles)

        if not mol:
            ecfp_fingerprints.append(None)
            rdkit_fingerprints.append(None)
            maccs_fingerprints.append(None)
            chem_phys_props.append(None)
            mhfp_encoder.append(None)
            erg_fingerprints.append(None)
            continue

        ecfp_fingerprints.append(mfpgen.GetFingerprint(mol))
        rdkit_fingerprints.append(rdkgen.GetFingerprint(mol))
        maccs_fingerprints.append(MACCSkeys.GenMACCSKeys(mol))
        mhfp6_fingerprints.append(mhfp_encoder.encode(can_smiles, radius=3))
        erg_fingerprints.append(rdReducedGraphs.GetErGFingerprint(mol))

        vals = Descriptors.CalcMolDescriptors(mol)

        chem_phys_props.append(
            {
                "slogp": round(vals["MolLogP"], 2),
                "smr": round(vals["MolMR"], 2),
                "labute_asa": round(vals["LabuteASA"], 2),
                "tpsa": round(vals["TPSA"], 2),
                "exact_mw": round(vals["ExactMolWt"], 2),
                "num_lipinski_hba": rdMolDescriptors.CalcNumLipinskiHBA(mol),
                "num_lipinski_hbd": rdMolDescriptors.CalcNumLipinskiHBD(mol),
                "num_rotatable_bonds": vals["NumRotatableBonds"],
                "num_hba": vals["NumHAcceptors"],
                "num_hbd": vals["NumHDonors"],
                "num_amide_bonds": rdMolDescriptors.CalcNumAmideBonds(mol),
                "num_heteroatoms": vals["NumHeteroatoms"],
                "num_heavy_atoms": vals["HeavyAtomCount"],
                "num_atoms": rdMolDescriptors.CalcNumAtoms(mol),
                "num_stereocenters": rdMolDescriptors.CalcNumAtomStereoCenters(mol),
                "num_unspecified_stereocenters": rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(
                    mol
                ),
                "num_rings": vals["RingCount"],
                "num_aromatic_rings": vals["NumAromaticRings"],
                "num_aliphatic_rings": vals["NumAliphaticRings"],
                "num_saturated_rings": vals["NumSaturatedRings"],
                "num_aromatic_heterocycles": vals["NumAromaticHeterocycles"],
                "num_aliphatic_heterocycles": vals["NumAliphaticHeterocycles"],
                "num_saturated_heterocycles": vals["NumSaturatedHeterocycles"],
                "num_aromatic_carbocycles": vals["NumAromaticCarbocycles"],
                "num_aliphatic_carbocycles": vals["NumAliphaticCarbocycles"],
                "num_saturated_carbocycles": vals["NumSaturatedCarbocycles"],
                "fraction_csp3": round(vals["FractionCSP3"], 2),
                "num_brdigehead_atoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                "bertz_complexity": GraphDescriptors.BertzCT(mol),
            }
        )

    smiles_df["ecfp4"] = ecfp_fingerprints
    smiles_df["rdkit"] = rdkit_fingerprints
    smiles_df["maccs"] = maccs_fingerprints
    smiles_df["chem_phys"] = chem_phys_props
    smiles_df["mhfp6"] = mhfp6_fingerprints
    smiles_df["erg"] = erg_fingerprints

    logger.info("🏃 Running model")

    smiles_df_subset = smiles_df.dropna(subset=[fingerprint_name])[
        ["smiles", fingerprint_name]
    ]
    if model_name == "rf":
        predictions = model.predict(smiles_df_subset[fingerprint_name].tolist())
        prediction_proba = model.predict_proba(
            smiles_df_subset[fingerprint_name].tolist()
        )
        label_classes = model.classes_.tolist()
    else:
        predictions = model.predict(smiles_df_subset[fingerprint_name].values)
        prediction_proba = model.predict_proba(
            smiles_df_subset[fingerprint_name].values
        )
        label_classes = model.classes_.tolist()

    logger.info("✅ Finished task")

    st.write("### Predictions")
    smiles_df_subset["Prediction"] = predictions
    probs = []
    for idx, probability in enumerate(prediction_proba):
        predicted_class = predictions[idx]
        probs.append(probability[label_classes.index(predicted_class)])
    smiles_df_subset["Probability"] = probs

    st.dataframe(smiles_df_subset[["smiles", "Prediction", "Probability"]])
