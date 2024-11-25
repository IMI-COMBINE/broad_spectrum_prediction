# Predicting antimicrobial class specificity of small molecules using machine learning

# Overview of Work
In our work, we developed large antimicrobial knowledge graphs (AntiMicrobial-KG) as repositories for collecting and visualizing public bioassay data involving in-vitro antibacterial assay - endpoints, such as Minimal Inhibitory Concentration (MIC). Utilizing the data in AntiMicrobial-KG, we build ML models to efficiently scan compound libraries to identify compounds with the potential to exhibit antibiotic activity. Moreover, our best model may assist in discriminating between broad-spectrum anti-bacterial compounds and compounds that are selectively active against Gram-positive, Gram-negative, Acid-fast, or Fungal pathogens. Our strategy involved training seven classic ML models across six compound fingerprint representations: ECFP8, MACCS, RDKit, ErG, MHFP6, and physicochemical properties. To counter imbalances in the dataset, which could affect model robustness, we performed a minority over-sampling technique, SMOTE, allowing for an increment in the performance of our models over existing ML models. Our investigation revealed that Random Forest trained on the MHFP6 fingerprint outperformed the other six models, demonstrating an accuracy of 75.9% and Cohen’s Kappa score of 0.68. Finally, we illustrated the model’s applicability for predicting antimicrobial compound properties of two small molecule screening libraries. Firstly, the EU-OpenScreen European Chemical Biology Library (ECBL) was tested against a panel of Gram-positive, Gram-negative, and Fungal pathogens. Here, we unveiled that the model was able to correctly predict more than 30% of active compounds for Gram-positive, Gram-negative, and Fungal pathogens. Secondly, with the Enamine library, a commercially available HTS compound collection with claimed antibacterial properties, we predicted its antimicrobial activity and pathogen class specificity. 

# Directory and file structure

The file and folder structure for the models processed through the training and hyperparameter optimization.
```bash
├── LICENSE
├── README.md
├── data
│   ├── benchmark
│   │   ├── bioactive_assay_results.csv
│   │   ├── ecbl_assay_results.csv
│   ├── chembl_34
│   │   └── chembl_34.db
│   ├── fingerprints
│   │   ├── acid-fast_chem_phys.tsv
│   │   ├── acid-fast_ecfp8.tsv
│   │   ├── acid-fast_erg.tsv
│   │   ├── acid-fast_maccs.tsv
│   │   ├── acid-fast_mhfp6.tsv
│   │   ├── acid-fast_rdkit.tsv
│   │   ├── combined_chem_phys.tsv
│   │   ├── combined_ecfp4.tsv
│   │   ├── combined_erg.tsv
│   │   ├── combined_maccs.tsv
│   │   ├── combined_mhfp6.tsv
│   │   ├── combined_rdkit.tsv
│   │   ├── fungi_chem_phys.tsv
│   │   ├── fungi_ecfp8.tsv
│   │   ├── fungi_erg.tsv
│   │   ├── fungi_maccs.tsv
│   │   ├── fungi_mhfp6.tsv
│   │   ├── fungi_rdkit.tsv
│   │   ├── gram-negative_chem_phys.tsv
│   │   ├── gram-negative_ecfp8.tsv
│   │   ├── gram-negative_erg.tsv
│   │   ├── gram-negative_maccs.tsv
│   │   ├── gram-negative_mhfp6.tsv
│   │   ├── gram-negative_rdkit.tsv
│   │   ├── gram-positive_chem_phys.tsv
│   │   ├── gram-positive_ecfp8.tsv
│   │   ├── gram-positive_erg.tsv
│   │   ├── gram-positive_maccs.tsv
│   │   ├── gram-positive_mhfp6.tsv
│   │   └── gram-positive_rdkit.tsv
│   ├── mapping
│   │   ├── bact_mapper.json
│   │   └── structure2class.json
│   ├── predictions
│   │   ├── combined_enamine_antibacterial_predictions.csv
│   │   ├── combined_test_metrics.tsv
│   │   ├── euos_biactive_predictions.csv
│   │   └── euos_ecbl_predictions.csv
│   ├── processed
│   │   ├── bacterial_chembl_34.tsv
│   │   ├── bacterial_coadd.tsv
│   │   ├── bacterial_spark.tsv
│   │   ├── chembl34_raw_data.tsv
│   │   ├── coadd_raw_data.tsv
│   │   ├── combined_bioassay_data.tsv
│   │   ├── combined_model_comparison_results.tsv
│   │   ├── full_data_with_fingerprints.pkl
│   │   └── spark_raw_data.tsv
│   ├── raw
│   │   ├── CO-ADD_DoseResponseData_r03_01-02-2020_CSV.zip
│   │   ├── SPARK Data Achaogen Contribution.csv
│   │   ├── SPARK Data CO-ADD Contribution.csv
│   │   ├── SPARK Data Compounds & Physicochemical Properties.csv
│   │   ├── SPARK Data Merck & Kyorin Contribution.csv
│   │   ├── SPARK Data Novartis Contribution.csv
│   │   ├── SPARK Data Quave Lab {Emory University} Publications.csv
│   │   ├── SPARK MIC Data.csv
│   └── splits
│       ├── acid-fast
│       │   ├── chem_phys_smote_train.csv
│       │   ├── chem_phys_test.csv
│       │   ├── chem_phys_train.csv
│       │   ├── ecfp8_smote_train.csv
│       │   ├── ecfp8_test.csv
│       │   ├── ecfp8_train.csv
│       │   ├── erg_smote_train.csv
│       │   ├── erg_test.csv
│       │   ├── erg_train.csv
│       │   ├── maccs_smote_train.csv
│       │   ├── maccs_test.csv
│       │   ├── maccs_train.csv
│       │   ├── mhfp6_smote_train.csv
│       │   ├── mhfp6_test.csv
│       │   ├── mhfp6_train.csv
│       │   ├── rdkit_smote_train.csv
│       │   ├── rdkit_test.csv
│       │   └── rdkit_train.csv
│       ├── combined
│       │   ├── chem_phys_smote_train.csv
│       │   ├── chem_phys_test.csv
│       │   ├── chem_phys_train.csv
│       │   ├── ecfp8_smote_train.csv
│       │   ├── ecfp8_test.csv
│       │   ├── ecfp8_train.csv
│       │   ├── erg_smote_train.csv
│       │   ├── erg_test.csv
│       │   ├── erg_train.csv
│       │   ├── maccs_smote_train.csv
│       │   ├── maccs_test.csv
│       │   ├── maccs_train.csv
│       │   ├── mhfp6_smote_train.csv
│       │   ├── mhfp6_test.csv
│       │   ├── mhfp6_train.csv
│       │   ├── rdkit_smote_train.csv
│       │   ├── rdkit_test.csv
│       │   └── rdkit_train.csv
│       ├── fungi
│       │   ├── chem_phys_smote_train.csv
│       │   ├── chem_phys_test.csv
│       │   ├── chem_phys_train.csv
│       │   ├── ecfp8_smote_train.csv
│       │   ├── ecfp8_test.csv
│       │   ├── ecfp8_train.csv
│       │   ├── erg_smote_train.csv
│       │   ├── erg_test.csv
│       │   ├── erg_train.csv
│       │   ├── maccs_smote_train.csv
│       │   ├── maccs_test.csv
│       │   ├── maccs_train.csv
│       │   ├── mhfp6_smote_train.csv
│       │   ├── mhfp6_test.csv
│       │   ├── mhfp6_train.csv
│       │   ├── rdkit_smote_train.csv
│       │   ├── rdkit_test.csv
│       │   └── rdkit_train.csv
│       ├── gram-negative
│       │   ├── chem_phys_smote_train.csv
│       │   ├── chem_phys_test.csv
│       │   ├── chem_phys_train.csv
│       │   ├── ecfp8_smote_train.csv
│       │   ├── ecfp8_test.csv
│       │   ├── ecfp8_train.csv
│       │   ├── erg_smote_train.csv
│       │   ├── erg_test.csv
│       │   ├── erg_train.csv
│       │   ├── maccs_smote_train.csv
│       │   ├── maccs_test.csv
│       │   ├── maccs_train.csv
│       │   ├── mhfp6_smote_train.csv
│       │   ├── mhfp6_test.csv
│       │   ├── mhfp6_train.csv
│       │   ├── rdkit_smote_train.csv
│       │   ├── rdkit_test.csv
│       │   └── rdkit_train.csv
│       └── gram-positive
│           ├── chem_phys_smote_train.csv
│           ├── chem_phys_test.csv
│           ├── chem_phys_train.csv
│           ├── ecfp8_smote_train.csv
│           ├── ecfp8_test.csv
│           ├── ecfp8_train.csv
│           ├── erg_smote_train.csv
│           ├── erg_test.csv
│           ├── erg_train.csv
│           ├── maccs_smote_train.csv
│           ├── maccs_test.csv
│           ├── maccs_train.csv
│           ├── mhfp6_smote_train.csv
│           ├── mhfp6_test.csv
│           ├── mhfp6_train.csv
│           ├── rdkit_smote_train.csv
│           ├── rdkit_test.csv
│           └── rdkit_train.csv
├── experiments
│   └── combined
│       ├── chem_phys_rf
│       │   ├── trial_0.pkl
│       │   ├── trial_0_params.json
│       │   ├── trial_1.pkl
│       │   ├── trial_10.pkl
│       │   ├── trial_10_params.json
│       │   ├── trial_11.pkl
│       │   ├── trial_11_params.json
│       │   ├── trial_12.pkl
│       │   ├── trial_12_params.json
│       │   ├── trial_13.pkl
│       │   ├── trial_13_params.json
│       │   ├── trial_14.pkl
│       │   ├── trial_14_params.json
│       │   ├── trial_1_params.json
│       │   ├── trial_2.pkl
│       │   ├── trial_2_params.json
│       │   ├── trial_3.pkl
│       │   ├── trial_3_params.json
│       │   ├── trial_4.pkl
│       │   ├── trial_4_params.json
│       │   ├── trial_5.pkl
│       │   ├── trial_5_params.json
│       │   ├── trial_6.pkl
│       │   ├── trial_6_params.json
│       │   ├── trial_7.pkl
│       │   ├── trial_7_params.json
│       │   ├── trial_8.pkl
│       │   ├── trial_8_params.json
│       │   ├── trial_9.pkl
│       │   └── trial_9_params.json
│       ├── chem_phys_xgboost
│       │   ├── trial_0.pickle.dat
│       │   ├── trial_0_params.json
│       │   ├── trial_1.pickle.dat
│       │   ├── trial_10.pickle.dat
│       │   ├── trial_10_params.json
│       │   ├── trial_11.pickle.dat
│       │   ├── trial_11_params.json
│       │   ├── trial_12.pickle.dat
│       │   ├── trial_12_params.json
│       │   ├── trial_13.pickle.dat
│       │   ├── trial_13_params.json
│       │   ├── trial_14.pickle.dat
│       │   ├── trial_14_params.json
│       │   ├── trial_1_params.json
│       │   ├── trial_2.pickle.dat
│       │   ├── trial_2_params.json
│       │   ├── trial_3.pickle.dat
│       │   ├── trial_3_params.json
│       │   ├── trial_4.pickle.dat
│       │   ├── trial_4_params.json
│       │   ├── trial_6.pickle.dat
│       │   ├── trial_6_params.json
│       │   ├── trial_7.pickle.dat
│       │   ├── trial_7_params.json
│       │   ├── trial_8.pickle.dat
│       │   ├── trial_8_params.json
│       │   ├── trial_9.pickle.dat
│       │   └── trial_9_params.json
│       ├── ecfp8_rf
│       │   ├── trial_0.pkl
│       │   ├── trial_0_params.json
│       │   ├── trial_1.pkl
│       │   ├── trial_10.pkl
│       │   ├── trial_10_params.json
│       │   ├── trial_11.pkl
│       │   ├── trial_11_params.json
│       │   ├── trial_12.pkl
│       │   ├── trial_12_params.json
│       │   ├── trial_13.pkl
│       │   ├── trial_13_params.json
│       │   ├── trial_14.pkl
│       │   ├── trial_14_params.json
│       │   ├── trial_1_params.json
│       │   ├── trial_2.pkl
│       │   ├── trial_2_params.json
│       │   ├── trial_3.pkl
│       │   ├── trial_3_params.json
│       │   ├── trial_4.pkl
│       │   ├── trial_4_params.json
│       │   ├── trial_5.pkl
│       │   ├── trial_5_params.json
│       │   ├── trial_6.pkl
│       │   ├── trial_6_params.json
│       │   ├── trial_7.pkl
│       │   ├── trial_7_params.json
│       │   ├── trial_8.pkl
│       │   ├── trial_8_params.json
│       │   ├── trial_9.pkl
│       │   └── trial_9_params.json
│       ├── ecfp8_xgboost
│       │   ├── trial_0.pickle.dat
│       │   ├── trial_0_params.json
│       │   ├── trial_1.pickle.dat
│       │   ├── trial_10.pickle.dat
│       │   ├── trial_10_params.json
│       │   ├── trial_11.pickle.dat
│       │   ├── trial_11_params.json
│       │   ├── trial_12.pickle.dat
│       │   ├── trial_12_params.json
│       │   ├── trial_14.pickle.dat
│       │   ├── trial_14_params.json
│       │   ├── trial_1_params.json
│       │   ├── trial_2.pickle.dat
│       │   ├── trial_2_params.json
│       │   ├── trial_3.pickle.dat
│       │   ├── trial_3_params.json
│       │   ├── trial_4.pickle.dat
│       │   ├── trial_4_params.json
│       │   ├── trial_6.pickle.dat
│       │   ├── trial_6_params.json
│       │   ├── trial_7.pickle.dat
│       │   ├── trial_7_params.json
│       │   ├── trial_8.pickle.dat
│       │   ├── trial_8_params.json
│       │   ├── trial_9.pickle.dat
│       │   └── trial_9_params.json
│       ├── erg_rf
│       │   ├── trial_0.pkl
│       │   ├── trial_0_params.json
│       │   ├── trial_1.pkl
│       │   ├── trial_10.pkl
│       │   ├── trial_10_params.json
│       │   ├── trial_11.pkl
│       │   ├── trial_11_params.json
│       │   ├── trial_12.pkl
│       │   ├── trial_12_params.json
│       │   ├── trial_13.pkl
│       │   ├── trial_13_params.json
│       │   ├── trial_14.pkl
│       │   ├── trial_14_params.json
│       │   ├── trial_1_params.json
│       │   ├── trial_2.pkl
│       │   ├── trial_2_params.json
│       │   ├── trial_3.pkl
│       │   ├── trial_3_params.json
│       │   ├── trial_4.pkl
│       │   ├── trial_4_params.json
│       │   ├── trial_5.pkl
│       │   ├── trial_5_params.json
│       │   ├── trial_6.pkl
│       │   ├── trial_6_params.json
│       │   ├── trial_7.pkl
│       │   ├── trial_7_params.json
│       │   ├── trial_8.pkl
│       │   ├── trial_8_params.json
│       │   ├── trial_9.pkl
│       │   └── trial_9_params.json
│       ├── erg_xgboost
│       │   ├── trial_0.pickle.dat
│       │   ├── trial_0_params.json
│       │   ├── trial_1.pickle.dat
│       │   ├── trial_10.pickle.dat
│       │   ├── trial_10_params.json
│       │   ├── trial_11.pickle.dat
│       │   ├── trial_11_params.json
│       │   ├── trial_12.pickle.dat
│       │   ├── trial_12_params.json
│       │   ├── trial_13.pickle.dat
│       │   ├── trial_13_params.json
│       │   ├── trial_14.pickle.dat
│       │   ├── trial_14_params.json
│       │   ├── trial_1_params.json
│       │   ├── trial_2.pickle.dat
│       │   ├── trial_2_params.json
│       │   ├── trial_3.pickle.dat
│       │   ├── trial_3_params.json
│       │   ├── trial_4.pickle.dat
│       │   ├── trial_4_params.json
│       │   ├── trial_5.pickle.dat
│       │   ├── trial_5_params.json
│       │   ├── trial_6.pickle.dat
│       │   ├── trial_6_params.json
│       │   ├── trial_9.pickle.dat
│       │   └── trial_9_params.json
│       ├── maccs_rf
│       │   ├── trial_0.pkl
│       │   ├── trial_0_params.json
│       │   ├── trial_1.pkl
│       │   ├── trial_10.pkl
│       │   ├── trial_10_params.json
│       │   ├── trial_11.pkl
│       │   ├── trial_11_params.json
│       │   ├── trial_12.pkl
│       │   ├── trial_12_params.json
│       │   ├── trial_13.pkl
│       │   ├── trial_13_params.json
│       │   ├── trial_14.pkl
│       │   ├── trial_14_params.json
│       │   ├── trial_1_params.json
│       │   ├── trial_2.pkl
│       │   ├── trial_2_params.json
│       │   ├── trial_3.pkl
│       │   ├── trial_3_params.json
│       │   ├── trial_4.pkl
│       │   ├── trial_4_params.json
│       │   ├── trial_5.pkl
│       │   ├── trial_5_params.json
│       │   ├── trial_6.pkl
│       │   ├── trial_6_params.json
│       │   ├── trial_7.pkl
│       │   ├── trial_7_params.json
│       │   ├── trial_8.pkl
│       │   ├── trial_8_params.json
│       │   ├── trial_9.pkl
│       │   └── trial_9_params.json
│       ├── maccs_xgboost
│       │   ├── trial_0.pickle.dat
│       │   ├── trial_0_params.json
│       │   ├── trial_1.pickle.dat
│       │   ├── trial_10.pickle.dat
│       │   ├── trial_10_params.json
│       │   ├── trial_11.pickle.dat
│       │   ├── trial_11_params.json
│       │   ├── trial_12.pickle.dat
│       │   ├── trial_12_params.json
│       │   ├── trial_13.pickle.dat
│       │   ├── trial_13_params.json
│       │   ├── trial_14.pickle.dat
│       │   ├── trial_14_params.json
│       │   ├── trial_1_params.json
│       │   ├── trial_2.pickle.dat
│       │   ├── trial_2_params.json
│       │   ├── trial_3.pickle.dat
│       │   ├── trial_3_params.json
│       │   ├── trial_4.pickle.dat
│       │   ├── trial_4_params.json
│       │   ├── trial_5.pickle.dat
│       │   ├── trial_5_params.json
│       │   ├── trial_8.pickle.dat
│       │   └── trial_8_params.json
│       ├── mhfp6_rf
│       │   ├── trial_0.pkl
│       │   ├── trial_0_params.json
│       │   ├── trial_1.pkl
│       │   ├── trial_10.pkl
│       │   ├── trial_10_params.json
│       │   ├── trial_11.pkl
│       │   ├── trial_11_params.json
│       │   ├── trial_12.pkl
│       │   ├── trial_12_params.json
│       │   ├── trial_13.pkl
│       │   ├── trial_13_params.json
│       │   ├── trial_14.pkl
│       │   ├── trial_14_params.json
│       │   ├── trial_1_params.json
│       │   ├── trial_2.pkl
│       │   ├── trial_2_params.json
│       │   ├── trial_3.pkl
│       │   ├── trial_3_params.json
│       │   ├── trial_4.pkl
│       │   ├── trial_4_params.json
│       │   ├── trial_5.pkl
│       │   ├── trial_5_params.json
│       │   ├── trial_6.pkl
│       │   ├── trial_6_params.json
│       │   ├── trial_7.pkl
│       │   ├── trial_7_params.json
│       │   ├── trial_8.pkl
│       │   ├── trial_8_params.json
│       │   ├── trial_9.pkl
│       │   └── trial_9_params.json
│       ├── mhfp6_xgboost
│       │   ├── trial_0.pickle.dat
│       │   ├── trial_0_params.json
│       │   ├── trial_1.pickle.dat
│       │   ├── trial_10.pickle.dat
│       │   ├── trial_10_params.json
│       │   ├── trial_11.pickle.dat
│       │   ├── trial_11_params.json
│       │   ├── trial_12.pickle.dat
│       │   ├── trial_12_params.json
│       │   ├── trial_13.pickle.dat
│       │   ├── trial_13_params.json
│       │   ├── trial_14.pickle.dat
│       │   ├── trial_14_params.json
│       │   ├── trial_1_params.json
│       │   ├── trial_2.pickle.dat
│       │   ├── trial_2_params.json
│       │   ├── trial_3.pickle.dat
│       │   ├── trial_3_params.json
│       │   ├── trial_4.pickle.dat
│       │   ├── trial_4_params.json
│       │   ├── trial_5.pickle.dat
│       │   ├── trial_5_params.json
│       │   ├── trial_7.pickle.dat
│       │   ├── trial_7_params.json
│       │   ├── trial_8.pickle.dat
│       │   └── trial_8_params.json
│       ├── rdkit_rf
│       │   ├── trial_0.pkl
│       │   ├── trial_0_params.json
│       │   ├── trial_1.pkl
│       │   ├── trial_10.pkl
│       │   ├── trial_10_params.json
│       │   ├── trial_11.pkl
│       │   ├── trial_11_params.json
│       │   ├── trial_12.pkl
│       │   ├── trial_12_params.json
│       │   ├── trial_13.pkl
│       │   ├── trial_13_params.json
│       │   ├── trial_14.pkl
│       │   ├── trial_14_params.json
│       │   ├── trial_1_params.json
│       │   ├── trial_2.pkl
│       │   ├── trial_2_params.json
│       │   ├── trial_3.pkl
│       │   ├── trial_3_params.json
│       │   ├── trial_4.pkl
│       │   ├── trial_4_params.json
│       │   ├── trial_5.pkl
│       │   ├── trial_5_params.json
│       │   ├── trial_6.pkl
│       │   ├── trial_6_params.json
│       │   ├── trial_7.pkl
│       │   ├── trial_7_params.json
│       │   ├── trial_8.pkl
│       │   ├── trial_8_params.json
│       │   ├── trial_9.pkl
│       │   └── trial_9_params.json
│       └── rdkit_xgboost
│           ├── trial_0.pickle.dat
│           ├── trial_0_params.json
│           ├── trial_1.pickle.dat
│           ├── trial_10.pickle.dat
│           ├── trial_10_params.json
│           ├── trial_11.pickle.dat
│           ├── trial_11_params.json
│           ├── trial_12.pickle.dat
│           ├── trial_12_params.json
│           ├── trial_13.pickle.dat
│           ├── trial_13_params.json
│           ├── trial_14.pickle.dat
│           ├── trial_14_params.json
│           ├── trial_1_params.json
│           ├── trial_2.pickle.dat
│           ├── trial_2_params.json
│           ├── trial_3.pickle.dat
│           ├── trial_3_params.json
│           ├── trial_4.pickle.dat
│           ├── trial_4_params.json
│           ├── trial_5.pickle.dat
│           ├── trial_5_params.json
│           ├── trial_6.pickle.dat
│           ├── trial_6_params.json
│           ├── trial_7.pickle.dat
│           └── trial_7_params.json
├── figures
│   ├── amrkg_chemspace.html
│   ├── figure_1.png
│   ├── figure_10.png
│   ├── figure_2b.png
│   ├── figure_3.png
│   ├── figure_4.png
│   ├── figure_5.png
│   ├── figure_6a.png
│   ├── figure_6b.png
│   ├── figure_7.png
│   ├── figure_8.png
│   ├── figure_9.png
│   ├── supplementary_figure_1.png
│   ├── supplementary_figure_2.png
│   ├── supplementary_figure_3.png
│   ├── supplementary_figure_4.png
│   └── supplementary_figure_5.png
├── install_tmap.sh
├── models
│   └── combined
│       ├── chem_phys_rf.pkl
│       ├── chem_phys_rf_params.json
│       ├── chem_phys_xgboost.pickle.dat
│       ├── chem_phys_xgboost_params.json
│       ├── ecfp8_rf.pkl
│       ├── ecfp8_rf_params.json
│       ├── ecfp8_xgboost.pickle.dat
│       ├── ecfp8_xgboost_params.json
│       ├── erg_rf.pkl
│       ├── erg_rf_params.json
│       ├── erg_xgboost.pickle.dat
│       ├── erg_xgboost_params.json
│       ├── maccs_rf.pkl
│       ├── maccs_rf_params.json
│       ├── maccs_xgboost.pickle.dat
│       ├── maccs_xgboost_params.json
│       ├── mhfp6_rf.pkl
│       ├── mhfp6_rf_params.json
│       ├── mhfp6_xgboost.pickle.dat
│       ├── mhfp6_xgboost_params.json
│       ├── rdkit_rf.pkl
│       ├── rdkit_rf_params.json
│       ├── rdkit_xgboost.pickle.dat
│       └── rdkit_xgboost_params.json
├── preprocess
│   ├── chembl_bacterial_activity.ipynb
│   ├── coadd_process.ipynb
│   ├── graph.py
│   ├── spark_process.ipynb
│   └── utils.py
├── requirements.txt
├── src
│   ├── 0_eda_analysis.ipynb
│   ├── 1_fingerprint_generator.ipynb
│   ├── 2_data_splitting.ipynb
│   ├── 3_amr_kg.ipynb
│   ├── 4_1_scaffold_diversity.ipynb
│   ├── 4_2_chem_class_analysis.ipynb
│   ├── 4_3_np_analysis.ipynb
│   ├── 5_model_selection.ipynb
│   ├── 6_model_training.ipynb
│   ├── 7_model_prediction.ipynb
│   ├── 8_feature_importance.ipynb
│   ├── 9_1_euos_library_prediction.ipynb
│   ├── 9_2_prediction_deep_dive.ipynb
│   ├── 9_3_cmpd_library_prediction.ipynb
│   ├── hpo_trainer.py
│   ├── logs.log
│   ├── model_selector.py
│   ├── predictor.py
│   └── utils.py
```

The notebooks in the (`src`)[src] folder include all the notebooks numbered in the form allowing you a step-by-step guide to have the models trained locally.

# Want to re-use our models?

It is great that you would like to benefit from our trained models. You can find the individual models on [Zenodo](https://zenodo.org/records/13868088) for local testing. Additionally, we have deployed the models online at https://antimicrobial-kg.serve.scilifelab.se/Model_Prediction. 

> [!IMPORTANT]  
> **We do not collect any SMILES submitted to our platform.**

# Funding

This work is part of the [COMBINE project](https://amr-accelerator.eu/project/combine/) that has received funding from the Innovative Medicines Initiative 2 Joint Undertaking under grant agreement No 853967. This Joint Undertaking receives support from the European Union’s Horizon 2020 research and innovation programme and EFPIA. 

The work reflects the author's view and that neither IMI nor the European Union, EFPIA, or any Associated Partners are responsible for any use that may be made of the information contained therein.


# More information
If you use our work, please consider citing us:
> will be updated soon
