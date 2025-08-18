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
        "figure.autolayout": True,
    })

my_theme()

# Title and description for dashboard
st.title("CO₂ Emissions Line Plot")
st.write("This plot shows CO₂ emission levels of countries from all over the world over time (from 1751-2014), with Canada highlighted in dark blue.")

# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 6))

# All countries
sns.lineplot(
    data = all_merged_drop[all_merged_drop["Indicator"] == "Emissions"],
    x = "Year",
    y = "Value",
    hue = "Country",
    estimator = None,
    alpha = 0.4,
    legend = False,
    ax = ax
)

# Canada highlighted
canada_data = all_merged_drop[
    (all_merged_drop["Indicator"] == "Emissions") &
    (all_merged_drop["Country"] == "Canada")
]
sns.lineplot(
    data = canada_data,
    x = "Year",
    y = "Value",
    color = "blue",
    label = "Canada",
    linewidth = 2.5,
    ax = ax
)

# Labels
ax.set_title("Country CO₂ Emissions per Year (1751–2014)")
ax.set_xlabel("Year")
ax.set_ylabel("Emissions (Metric Tonnes)")
ax.legend()

# Show in Streamlit
st.pyplot(fig)
