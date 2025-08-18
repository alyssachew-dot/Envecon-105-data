!pip install plotnine
!pip install adjustText
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import statsmodels.formula.api as smf
import seaborn as sns
from adjustText import adjust_text

# -----------------------
# Title and description
# -----------------------
st.title("Alyssa Chew Envecon 105 Dashboard")
st.write("""This is a simple dashboard demonstrating interactive widgets, charts, and layout using Streamlit.""")


all_merged_drop = pd.read_csv("https://raw.githubusercontent.com/alyssachew-dot/Envecon-105-data/refs/heads/main/all_merged_drop.csv")
def my_theme():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 16,
        "axes.titleweight": "normal",
        "figure.autolayout": True,})

my_theme()

plt.figure(figsize=(12, 6))
sns.lineplot(
    data = all_merged_drop[all_merged_drop["Indicator"] == "Emissions"],
    x = "Year",
    y = "Value",
    hue = "Country",
    estimator = None,
    alpha=0.4,
    legend=False)
canada_data = all_merged_drop[(all_merged_drop["Indicator"] == "Emissions") & (all_merged_drop["Country"] == "Canada")]
sns.lineplot(
    data = canada_data,
    x="Year",
    y="Value",
    color="blue",
    label="Canada",
    linewidth=2.5)

plt.title("Country CO\u2082 Emissions per Year (1751-2014)")
plt.xlabel("Year")
plt.ylabel("Emissions (Metric Tonnes)")

plt.legend()
plt.show()

  

