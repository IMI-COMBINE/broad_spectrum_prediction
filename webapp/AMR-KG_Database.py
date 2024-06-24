import pandas as pd
from collections import defaultdict
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    layout="wide",
    page_title="AMR-KG",
    page_icon=":microscope:",
    initial_sidebar_state="auto",
)

st.markdown(
    "<h1 style='text-align: center; color: #78bc1e;'>AMR-KG Database</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align: center;'>An exhaustive data warehouse of experimentally \
    validated antibacterial chemicals</h4>",
    unsafe_allow_html=True,
)

# Add some styling with CSS selectors
st.markdown(
    """
    <style>

    a[href] {
        color: #1e85bc;
    }

    [data-testid="column"]:nth-child(1){background-color: #78bc1e;}
    [data-testid="column"]:nth-child(2){background-color: #78bc1e;}
    [data-testid="column"]:nth-child(3){background-color: #78bc1e;}

    </style>
    """,
    unsafe_allow_html=True,
)

# AMR-KG Description
st.header(
    "ℹ️ About the resources",
    divider="orange",
    help="Information on the data in AMR-KG.",
)

st.write(
    "Antimicrobial Resistant Knowledge Graph (AMR-KG) is an exhaustive data warehouse of experimentally validated antibacterial chemicals \
    covering Gram-positive, Gram-negative, acid-fast bacteria and fungi. The construction of the AMR-KG involved collecting \
    minimum inhibitory concentration (MIC) data from three different public data resources:"
)


col = st.columns(3)

with col[0]:
    container = st.container(border=True, height=280)
    container.write("### [ChEMBL](https://www.ebi.ac.uk/chembl/)")

    container.write(
        "ChEMBL is a manually curated database of bioactive molecules with drug-like properties. \
        It brings together chemical and bioactivity to aid the translation of experimental information into effective new drugs.",
    )

with col[1]:
    container = st.container(border=True, height=280)
    container.write("### [CO-ADD](https://co-add.org/)")
    container.write(
        "Community for Open Antimicrobial Drug Discovery (CO-ADD) is a not-for-profit initiative led by academics at The University of Queensland. \
        It provides free antimicrobial screening for researchers worldwide."
    )

with col[2]:
    container = st.container(border=True, height=280)
    container.write("### [SPARK](spark.co-add.org)")
    container.write(
        "Shared Platform for Antibiotic Research (SPARK), now integrated and maintained by the CO-ADD community, was initially created by the\
        Pew Charitable Trusts to expand research around antibiotics targeting Gram-negative bacteria."
    )


# """Stats about the data"""
st.header(
    "📊 Data overview",
    divider="orange",
    help="Stats on the underlying data.",
)
DATA_DIR = "../data"
df = pd.read_csv(f"{DATA_DIR}/processed/combined_bioassay_data.tsv", sep="\t")


def get_base_stats():

    chembl_cmpds = set(
        df[df["compound_source"] == "chembl_34"]["compound_inchikey"].unique()
    )
    coadd_cmpds = set(
        df[df["compound_source"] == "coadd_03_01-02-2020"]["compound_inchikey"].unique()
    )
    spark_cmpds = set(
        df[df["compound_source"] == "spark"]["compound_inchikey"].unique()
    )

    pchem_dist_dict = defaultdict(list)

    for idx, row in df.iterrows():
        (
            inchikey,
            smiles,
            source,
            gram_pos,
            gram_neg,
            fungi,
            acid_fast,
            _,
            _,
            _,
            best_class,
        ) = row

        if best_class == "gram-positive":
            pchem_dist_dict["gram-positive"].append(gram_pos)

        elif best_class == "gram-negative":
            pchem_dist_dict["gram-negative"].append(gram_neg)

        elif best_class == "fungi":
            pchem_dist_dict["fungi"].append(fungi)

        elif best_class == "acid-fast":
            pchem_dist_dict["acid-fast"].append(acid_fast)

    return (chembl_cmpds, coadd_cmpds, spark_cmpds), pchem_dist_dict


(chembl_cmpds, coadd_cmpds, spark_cmpds), pchem_dist_dict = get_base_stats()

fig = plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.bar(
    ["ChEMBL", "CO-ADD", "SPARK"],
    [len(chembl_cmpds), len(coadd_cmpds), len(spark_cmpds)],
)
plt.title("Compound distribution across resources")
plt.yscale("log")
plt.ylabel("Number of compounds")

plt.subplot(2, 2, 2)
plt.hist(
    pchem_dist_dict["gram-positive"], alpha=0.5, color="red", label="gram-positive"
)
plt.hist(
    pchem_dist_dict["gram-negative"], alpha=0.5, color="blue", label="gram-negative"
)
plt.hist(pchem_dist_dict["fungi"], alpha=0.5, color="green", label="fungi")
plt.hist(pchem_dist_dict["acid-fast"], alpha=0.5, color="orange", label="acid-fast")
plt.legend()
plt.title("Distribution of pChEMBL values")
plt.ylabel("Number of compounds")
plt.xlabel("pChEMBL value")

st.pyplot(fig)


st.header(
    ":arrow_down: Data Download",
    divider="orange",
    help="Downloading all the data in the AMR-KG.",
)


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False, sep="\t").encode("utf-8")


csv = convert_df(df)
st.write("The data files contains the following columns in the tab-seperated manner:")
st.dataframe(df.head(3))
st.download_button(
    "Press to Download", csv, "amrkg_data_dump.tsv", "text/tsv", key="download-tsv"
)

# Publucation note
# with st.expander("Check out our publication for more deatils "):
# st.write(
#     """"""
# )
