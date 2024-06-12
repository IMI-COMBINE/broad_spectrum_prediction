"""Python script with for generation of property Graph."""

import pandas as pd
from tqdm import tqdm
from ast import literal_eval

from neomodel import (
    config,
    StructuredNode,
    StringProperty,
    RelationshipTo,
    StructuredRel,
    FloatProperty,
    ArrayProperty,
    db,
)

from neo4j import GraphDatabase


def connect_db():
    """Connect to the Neo4j database."""
    # This requires a local community version installation of Neo4j
    my_driver = GraphDatabase().driver(
        "bolt://localhost:7687", auth=("user", "administrator")
    )
    config.DRIVER = my_driver

    # Delete all nodes and relationships
    db.cypher_query("MATCH ()-[r]-() DELETE r")  # delete all relationships
    db.cypher_query("MATCH (n) DETACH DELETE n")  # delete all nodes


# Predifinding of node and relationship classes
class Activity(StructuredRel):
    act_type = StringProperty()
    relation = StringProperty()
    value = FloatProperty()
    unit = StringProperty()
    pmic = FloatProperty()


# Defining the nodes
class BacteriaStrain(StructuredNode):
    name = StringProperty(required=True, unique_index=True, unique=True)


class Bacteria(StructuredNode):
    name = StringProperty(required=True, unique_index=True, unique=True)

    # outgoing relations
    strain = RelationshipTo(BacteriaStrain, "IS_A")


class Chemical(StructuredNode):
    name = StringProperty(required=True, unique_index=True, unique=True)
    smiles = StringProperty(index=True)
    inchikey = StringProperty(index=True)
    chem_class = ArrayProperty(StringProperty())
    chem_superclass = ArrayProperty(StringProperty())
    chem_pathway = ArrayProperty(StringProperty())

    # outgoing relations
    activity = RelationshipTo(Bacteria, "SHOWS_ACTIVITY_ON", model=Activity)


def load_data():
    """Load the data from the processed files."""
    DATA_DIR = "../data"

    """ChEMBL data"""
    chembl_df = pd.read_csv(
        f"{DATA_DIR}/processed/chembl33_raw_data.tsv", sep="\t", low_memory=False
    )
    chembl_df.rename(
        columns={
            "pref_name": "compound_name",
            "chembl_id": "compound_id",
            "canonical_smiles": "smiles",
            "standard_inchi_key": "inchikey",
            "ORGANISM": "organism",
        },
        inplace=True,
    )
    chembl_df.drop(
        columns=[
            "standard_relation",
            "standard_type",
            "standard_inchi",
            "STRAIN",
            "assay_tax_id",
        ],
        inplace=True,
    )
    chembl_df["compound_id"] = chembl_df["compound_id"].apply(
        lambda x: f"CHEMBL:CHEMBL{x}"
    )

    """CO-ADD data"""
    coadd_df = pd.read_csv(f"{DATA_DIR}/processed/coadd_raw_data.tsv", sep="\t")
    coadd_df.rename(
        columns={
            "COMPOUND_NAME": "compound_name",
            "COADD_ID": "compound_id",
            "SMILES": "smiles",
            "compound_inchikey": "inchikey",
            "DRVAL_MEDIAN": "standard_value",
            "DRVAL_UNIT": "standard_units",
            "ORGANISM": "organism",
        },
        inplace=True,
    )
    coadd_df.drop(columns=["DRVAL_TYPE"], inplace=True)

    """SPARK data"""
    spark_df = pd.read_csv(f"{DATA_DIR}/processed/spark_raw_data.tsv", sep="\t")
    spark_df.rename(
        columns={
            "compound_name": "compound_id",
            "canonical_smiles": "smiles",
            "compound_inchikey": "inchikey",
        },
        inplace=True,
    )
    spark_df.drop(columns=["standard_relation"], inplace=True)
    spark_df["compound_id"] = spark_df["compound_id"].apply(lambda x: f"SPARK:{x}")

    combined_df = pd.concat([chembl_df, coadd_df, spark_df], ignore_index=True)
    combined_df.drop_duplicates(inplace=True)

    # Adding in the chemical class, superclass and pathway
    all_fingerprint_data = pd.read_pickle(
        f"{DATA_DIR}/processed/full_data_with_fingerprints.pkl"
    )
    all_fingerprint_data = all_fingerprint_data[
        [
            "compound_inchikey",
            "chemical_class",
            "compound_superclass",
            "compound_pathway",
        ]
    ]

    combined_df = pd.merge(
        combined_df,
        all_fingerprint_data,
        left_on="inchikey",
        right_on="compound_inchikey",
        how="left",
    )

    return combined_df


if __name__ == "__main__":
    connect_db()

    df = load_data()

    # Compound nodes
    compound_nodes = df[
        [
            "compound_name",
            "compound_id",
            "smiles",
            "inchikey",
            "chemical_class",
            "compound_superclass",
            "compound_pathway",
        ]
    ].drop_duplicates()
    compound_data = []

    for _, compound_id, smiles, inchikey, classes, superclasses, pathways in tqdm(
        compound_nodes.values, desc="Generating nodes"
    ):
        if pd.isna(classes):
            classes = []
        else:
            classes = literal_eval(classes)

        if pd.isna(superclasses):
            superclasses = []
        else:
            superclasses = literal_eval(superclasses)

        if pd.isna(pathways):
            pathways = []
        else:
            pathways = literal_eval(pathways)

        c = Chemical(
            name=compound_id,
            smiles=smiles,
            inchikey=inchikey,
            chem_class=classes,
            chem_superclass=superclasses,
            chem_pathway=pathways,
        ).save()

    # Bacteria nodes
    bact_nodes = df["organism"].unique().tolist()
    for bact_name in tqdm(bact_nodes):
        b = Bacteria(name=bact_name).save()

    # Bacteria strain nodes
    for bact_strain_name in tqdm(df["bact_class"].unique()):
        b = BacteriaStrain(name=bact_strain_name).save()

    # Connecting the nodes based on schema
    for row in tqdm(df.values, desc="Generating relationships"):
        (
            _,
            cmp_id,
            cmp_smiles,
            cmp_inchikey,
            value,
            unit,
            bact,
            assay_id,
            bact_class,
            pmic,
            _,
            _,
            _,
            _,
        ) = row
        c = Chemical.nodes.get(name=cmp_id)
        b = Bacteria.nodes.get(name=bact)
        s = BacteriaStrain.nodes.get(name=bact_class)

        c.activity.connect(
            b,
            {
                "act_type": "MIC",
                "relation": "=",
                "value": value,
                "unit": unit,
                "pmic": pmic,
            },
        )

        b.strain.connect(s)
