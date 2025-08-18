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
    legend=False,
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
st.write("This plot shows CO₂ emission levels of countries from all over the world over time (from 1751-2014), the left plot showing the US highlighted and the right plot showing Canada highlighted both in dark blue.")

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




# top 10 line plot with country labels 

from plotnine import (
    ggplot, aes, geom_line, geom_text, labs, theme_classic,
    theme, element_text, scale_x_continuous
)
from plotnine.scales import scale_color_cmap_d
import pandas as pd
import numpy as np

top_10_count = (
    all_merged_drop[
        (all_merged_drop["Indicator"] == "Emissions") &
        (all_merged_drop["Year"] == 2014)
    ].copy()
)
top_10_count["rank"] = top_10_count["Value"].rank(method="dense", ascending=False).astype(int)
top_10_count = top_10_count[top_10_count["rank"] <= 10].sort_values("rank")
top_10_countries = top_10_count["Country"].tolist()

Top10b = all_merged_drop[
    (all_merged_drop["Country"].isin(top_10_countries)) &
    (all_merged_drop["Indicator"] == "Emissions") &
    (all_merged_drop["Year"] >= 1900)
].copy()

labels_df = Top10b.groupby("Country").apply(
    lambda df: df[df["Year"] == df["Year"].max()]
).reset_index(drop=True)

labels_df = labels_df.sort_values("Value").reset_index(drop=True)
min_dist = 0.03 * (labels_df["Value"].max() - labels_df["Value"].min())
y_positions = labels_df["Value"].values.copy()
for i in range(1, len(y_positions)):
    if y_positions[i] - y_positions[i-1] < min_dist:
        y_positions[i] = y_positions[i-1] + min_dist
labels_df["y_spread"] = y_positions

Top10b_plot = (
    ggplot(Top10b, aes(x="Year", y="Value", color="Country"))
    + geom_line(size=1)
    + geom_text(
        data=labels_df,
        mapping=aes(x="Year", y="y_spread", label="Country"),
        ha="left",
        va="center",
        nudge_x=0.5
    )
    + scale_color_cmap_d(cmap_name="viridis")
    + scale_x_continuous(
        breaks=range(1900, 2015, 5),
        expand=(0.05, 0)
    )
    + labs(
        title="Top 10 Emissions-producing Countries in 2010 (1900–2014)",
        subtitle="Ordered by Emissions Produced in 2014",
        x="Year",
        y="Emissions (Metric Tons)"
    )
    + theme_classic()
    + theme(
        figure_size=(12, 6),
        axis_text_x=element_text(size=12, rotation=45, ha="right"),
        axis_text_y=element_text(size=12),
        axis_title_x=element_text(size=14),
        axis_title_y=element_text(size=14),
        plot_title=element_text(size=16),
        legend_position="none"
    )
)
fig3 = Top10b_plot.draw()
st.subheader("Top 10 Emissions-Producing Countries (1900–2014)")
st.header("Top 10 CO₂ Emission-Producing Countries")
st.write("This line graph shows emissions for the top 10 countries ranked by 2014 emissions.")
st.pyplot(fig3)




# Distributions of indicators by Year and Value (USA)
from plotnine import (
    ggplot, aes, geom_line, facet_grid, labs, theme, element_text
)

filtered_df = all_merged_drop[
    ~all_merged_drop["Indicator"].isin(["Disasters", "Temperature"])
].copy()

distrib_indicators_plot = (
    ggplot(filtered_df, aes(x="Year", y="Value", group="Country"))
    + geom_line()
    + facet_grid("Indicator ~ Region", scales="free_y")
    + labs(
        title="Distribution of Indicators by Year and Value",
        y="Indicator Value"
    )
    + my_theme()  
    + theme(strip_text=element_text(size=16, face="bold"), figure_size=(12, 12))
)
fig4 = distrib_indicators_plot.draw()

# Distributions of indicators by Year and Value (Canada)
from plotnine import (
    ggplot, aes, geom_line, facet_grid, labs, theme, element_text
)

filtered_df = all_merged_drop_ca[
    ~all_merged_drop_ca["Indicator"].isin(["Disasters", "Temperature"])
].copy()

distrib_indicators_plot_ca = (
    ggplot(filtered_df, aes(x="Year", y="Value", group="Country"))
    + geom_line()
    + facet_grid("Indicator ~ Region", scales="free_y")
    + labs(
        title="Distribution of Indicators by Year and Value",
        y="Indicator Value"
    )
    + my_theme()  
    + theme(strip_text=element_text(size=16, face="bold"), figure_size=(12, 12))
)
fig5 = distrib_indicators_plot_ca.draw()


col1, col2 = st.columns(2)

st.header("Distribution of indicators (energy, GDP, and emissions) around the world and USA and Canada isoalted") 
st.write("Side-by-side display of the graphs of the world wide distribution of indicators and the graphs of just USA (left) and Canada (right) distribtions of indicators isolated")

with col1:
    st.subheader("USA line isolated")
    st.pyplot(fig4)

with col2:
    st.subheader("Canada line isolated")
    st.pyplot(fig5)





#facet plot (US)

CO2_temp_US = all_merged_drop[
    (all_merged_drop['Country'] == "United States") &
    (all_merged_drop['Year'] >= 1980) &
    (all_merged_drop['Year'] <= 2014) &
    (all_merged_drop['Indicator'].isin(["Emissions", "Temperature"]))].copy()


if 'Label' not in CO2_temp_US.columns:
    CO2_temp_US['Label'] = CO2_temp_US['Indicator'] 


CO2_temp_US['Label'] = pd.Categorical(CO2_temp_US['Label'], ordered=True)

CO2_temp_US_facet = (
    ggplot(CO2_temp_US, aes(x ='Year', y = 'Value'))
    + geom_point()
    + geom_smooth(method = 'loess', se = False,color = 'blue')
    + scale_x_continuous(breaks = range(1980, 2015, 5), labels = range(1980, 2015, 5))
    + facet_wrap('~Label', scales='free_y', ncol=1)
    + theme_classic()
    + theme(
        axis_text_x = element_text(size = 12, angle = 90, color = 'black'),
        axis_text_y = element_text(size = 12, color = 'black'),
        strip_text_x = element_text(size = 14),
        axis_title = element_blank(),
        plot_title = element_text(size = 16))
    + labs(title = "US Emissions and Temperatures (1980-2014)"))
fig6 = CO2_temp_US_facet.draw()


#facet plot (Canada)
CO2_temp_Canada = all_merged_drop[
    (all_merged_drop['Country'] == "Canada") &
    (all_merged_drop['Year'] >= 1980) &
    (all_merged_drop['Year'] <= 2014) &
    (all_merged_drop['Indicator'].isin(["Emissions", "Temperature"]))].copy()


if 'Label' not in CO2_temp_Canada.columns:
    CO2_temp_Canada['Label'] = CO2_temp_Canada['Indicator'] 


CO2_temp_Canada['Label'] = pd.Categorical(CO2_temp_Canada['Indicator'], categories = ["Emissions", "Temperature"], ordered=True)

CO2_temp_Canada_facet = (
    ggplot(CO2_temp_Canada, aes(x = 'Year', y = 'Value'))
    + geom_point()
    + geom_smooth(method = 'loess', se = False, color = 'blue')
    + scale_x_continuous(breaks = range(1980, 2015, 5), labels = range(1980, 2015, 5))
    + facet_wrap('~Label', scales='free_y', ncol=1)
    + theme_classic()
    + theme(
        axis_text_x = element_text(size = 12, angle = 90, color = 'black'),
        axis_text_y = element_text(size = 12, color = 'black'),
        strip_text_x = element_text(size = 14),
        axis_title = element_blank(),
        plot_title = element_text(size = 16))
    + labs(title = "Canada Emissions and Temperatures (1980-2014)"))

CO2_temp_Canada["Label"] = CO2_temp_Canada["Indicator"].map({
    "Emissions": "CO₂ Emissions (Metric Tonnes)",
    "Temperature": "Temperature departure (Celsius)"
})
fig7 = CO2_temp_Canada_facet.draw()

col1, col2 = st.columns(2)

st.header("USA and Canada Emissions and temperature scatterplots") 
st.write("Simultaneous display of the scatterplot of emissions v. time and temperature v. time for both the US (left) and Canada (right)")

with col1:
    st.subheader("USA specific graphs")
    st.pyplot(fig6)

with col2:
    st.subheader("Canada specific graphs")
    st.pyplot(fig7)




