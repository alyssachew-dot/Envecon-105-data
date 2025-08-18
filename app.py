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


all_merged_drop_ca = pd.read_csv("https://raw.githubusercontent.com/alyssachew-dot/Envecon-105-data/refs/heads/main/all_merged_drop.csv")
all_merged_drop = pd.read_csv("https://raw.githubusercontent.com/alyssachew-dot/Envecon-105-data/refs/heads/main/all_merged_drop_USA.csv")

# first plot (USA)
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

fig1, ax = plt.subplots(figsize=(12, 6))

sns.lineplot(
    data = all_merged_drop[all_merged_drop["Indicator"] == "Emissions"],
    x="Year",
    y="Value",
    hue="Country",
    estimator=None,
    alpha=0.4,
    legend=False'
    ax = ax)

us_data = all_merged_drop[(all_merged_drop["Indicator"] == "Emissions") & (all_merged_drop["Country"] == "United States")]
sns.lineplot(
    data=us_data,
    x="Year",
    y="Value",
    color="blue",
    label="United States",
    linewidth=2.5,
    ax = ax
)

plt.title("Country CO\u2082 Emissions per Year (1751-2014)")
plt.xlabel("Year")
plt.ylabel("Emissions (Metric Tonnes)")

plt.legend()
plt.show()


# first plot (canada)
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
fig2, ax = plt.subplots(figsize=(12, 6))

# All countries
sns.lineplot(
    data = all_merged_drop_ca[all_merged_drop_ca["Indicator"] == "Emissions"],
    x = "Year",
    y = "Value",
    hue = "Country",
    estimator = None,
    alpha = 0.4,
    legend = False,
    ax = ax
)

# Canada highlighted
canada_data = all_merged_drop_ca[
    (all_merged_drop_ca["Indicator"] == "Emissions") &
    (all_merged_drop_ca["Country"] == "Canada")
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



col1, col2 = st.columns(2)

with col1:
    st.subheader("USA Highlighted")
    st.pyplot(fig1)

with col2:
    st.subheader("Canada Highlighted")
    st.pyplot(fig2)




# heatmap
from plotnine import *
top_10 = ( all_merged_drop[
        (all_merged_drop["Indicator"] == "Emissions") & 
        (all_merged_drop["Year"] == 2014)]
    .assign(rank=lambda df: df["Value"].rank(method="dense", ascending=False).astype(int))
    .query("rank <= 10")
    .sort_values("rank"))

data = all_merged_drop[
    (all_merged_drop["Country"].isin(top_10["Country"])) &
    (all_merged_drop["Indicator"] == "Emissions") &
    (all_merged_drop["Year"] >= 1900)].copy()

order = data[data["Year"] == 2014].set_index("Country")["Value"].sort_values(ascending=False).index
data["Country"] = pd.Categorical(data["Country"], categories=order, ordered=True)

data["logValue"] = np.log(data["Value"])

plot = ( ggplot(data, aes("Year", "Country", fill="logValue"))
    + geom_tile()
    + scale_fill_cmap(name="viridis")
    + scale_x_continuous(breaks = range(1900, 2015, 5), labels = [str(y) for y in range(1900, 2015, 5)])
    + labs(
        title=r"Top 10 CO$_2$ Emission-producing Countries",
        subtitle = "Ranked by Emissions in 2014",
        fill = "Ln(CO2 Emissions)")
    + theme_classic()
    + theme(
        axis_text_x = element_text(size = 12, angle = 90, color = "black"),
        axis_text_y = element_text(size = 12, color = "black"),
        axis_title = element_blank(),
        plot_title = element_text(size = 16),
        legend_position = "bottom"))

fig = plot.draw()
st.header("Top 10 CO₂ Emission-Producing Countries")
st.write("This heatmap shows emissions for the top 10 countries ranked by 2014 emissions.")
st.pyplot(fig)
