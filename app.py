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


st.header("Distribution of indicators (energy, GDP, and emissions) around the world and USA and Canada isoalted") 
st.write("Side-by-side display of the graphs of the world wide distribution of indicators and the graphs of just USA (left) and Canada (right) distribtions of indicators isolated")

col1, col2 = st.columns(2)
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
CO2_temp_Canada = all_merged_drop_ca[
    (all_merged_drop_ca['Country'] == "Canada") &
    (all_merged_drop_ca['Year'] >= 1980) &
    (all_merged_drop_ca['Year'] <= 2014) &
    (all_merged_drop_ca['Indicator'].isin(["Emissions", "Temperature"]))].copy()


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


st.header("USA and Canada Emissions and temperature scatterplots") 
st.write("Simultaneous display of the scatterplot of emissions v. time and temperature v. time for both the US (left) and Canada (right)")
col1, col2 = st.columns(2)

with col1:
    st.subheader("USA specific graphs")
    st.pyplot(fig6)

with col2:
    st.subheader("Canada specific graphs")
    st.pyplot(fig7)



# scatterplots (US)
filtered = all_merged_drop[
    (all_merged_drop['Country'] == "United States") &
    (all_merged_drop['Year'] >= 1980) &
    (all_merged_drop['Year'] <= 2014)
]

filtered = filtered.drop(columns=['Label'])
wide_US = filtered.pivot(index='Year', columns='Indicator', values='Value').reset_index()

# Create a Matplotlib figure
fig8, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(data=wide_US, x='Emissions', y='Temperature', ax=ax)
sns.regplot(data=wide_US, x='Emissions', y='Temperature',
            ci=None, scatter=False, color="red", ax=ax)

ax.set_title("US Emissions and Temperature (1980–2014)", fontsize=16)
ax.set_xlabel("Emissions (Metric Tonnes)", fontsize=14)
ax.set_ylabel("Temperature (Fahrenheit)", fontsize=14)
ax.tick_params(axis="x", labelsize=12, colors="black")
ax.tick_params(axis="y", labelsize=12, colors="black")




# scatterplots (Canada)
# Filter and reshape data
filtered = all_merged_drop_ca[
    (all_merged_drop_ca['Country'] == "Canada") &
    (all_merged_drop_ca['Year'] >= 1980) &
    (all_merged_drop_ca['Year'] <= 2014)
]
filtered = filtered.drop(columns=['Label'])
wide_Canada = filtered.pivot(index='Year', columns='Indicator', values='Value').reset_index()

# Create figure
fig9, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(data=wide_Canada, x='Emissions', y='Temperature', color="black", s=40, ax=ax)
sns.regplot(data=wide_Canada, x='Emissions', y='Temperature',
            ci=None, scatter=False, color="red", ax=ax)

ax.set_title("Canada CO₂ Emissions and Temperature (1980–2014)", fontsize=16)
ax.set_xlabel("Emissions (Metric Tonnes)", fontsize=14)
ax.set_ylabel("Temperature Departure (Fahrenheit)", fontsize=14)
ax.tick_params(axis="x", labelsize=12, colors="black")
ax.tick_params(axis="y", labelsize=12, colors="black")


st.header("USA and Canada Emissions v. Temperature scatterplots") 
st.write("Scatterplot of Emissions v. Temperature for both the US (left) and Canada (right) with line of best fit")
col1, col2 = st.columns(2)

with col1:
    st.subheader("US Emissions vs Temperature (1980–2014)")
    st.pyplot(fig8)

with col2:
    st.subheader("Canada CO₂ Emissions vs Temperature (1980–2014)")
    st.pyplot(fig9)


# scatterplot z-score (USA) 
from scipy.stats import zscore
fig12, ax = plt.subplots(figsize=(8,6))

wide_US['Emissions_scaled'] = zscore(wide_US['Emissions'])
wide_US['Temperature_scaled'] = zscore(wide_US['Temperature'])
sns.regplot(
    data=wide_US,
    x='Emissions_scaled',
    y='Temperature_scaled',
    ci=None,
    color='black',
    line_kws={'color': 'blue'},
    scatter_kws={'color': 'black', 's': 40},
    ax=ax
)

ax.set_title("US CO₂ Emissions and Temperature (1980–2014)", fontsize=16)
ax.set_xlabel("Scaled Emissions (z-score)", fontsize=14)
ax.set_ylabel("Scaled Temperature (z-score)", fontsize=14)
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.grid(True)



# scatterplot z-score (Canada) 
fig13, ax = plt.subplots(figsize=(8,6))

wide_Canada['Emissions_scaled'] = zscore(wide_Canada['Emissions'])
wide_Canada['Temperature_scaled'] = zscore(wide_Canada['Temperature'])
sns.regplot(
    data=wide_Canada,
    x='Emissions_scaled',
    y='Temperature_scaled',
    ci=None,
    color='black',
    line_kws={'color': 'blue'},
    scatter_kws={'color': 'black', 's': 40},
    ax=ax
)

ax.set_title("Canada CO₂ Emissions and Temperature (1980–2014)", fontsize=16)
ax.set_xlabel("Scaled Emissions (z-score)", fontsize=14)
ax.set_ylabel("Scaled Temperature (z-score)", fontsize=14)
ax.tick_params(axis="x", labelsize=12, colors="black")
ax.tick_params(axis="y", labelsize=12, colors="black")
ax.grid(True)



st.header("USA and Canada Emissions v. Temperature (standardized) scatterplots") 
st.write("Scatterplot of Emissions v.  Temperature (standardized) for both the US (left) and Canada (right) with line of best fit")
col1, col2 = st.columns(2)

with col1:
    st.subheader("USA CO₂ Emissions vs Temperature (standardized) (1980–2014)")
    st.pyplot(fig12)

with col2:
    st.subheader("Canada CO₂ Emissions vs Temperature (standardized) (1980–2014)")
    st.pyplot(fig13)


# disaster v emissions scatterplot (USA)
filtered = all_merged_drop[
    (all_merged_drop['Country'] == "United States") &
    (all_merged_drop['Year'] >= 1980) &
    (all_merged_drop['Year'] <= 2014)
]
filtered = filtered.drop(columns=['Label'])
wide_US = filtered.pivot(index='Year', columns='Indicator', values='Value').reset_index()

# Plot
fig10, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=wide_US, x='Emissions', y='Disasters', color="black", s=40, ax=ax)
sns.regplot(data=wide_US, x='Emissions', y='Disasters', ci=None, scatter=False, color="red", ax=ax)

ax.set_title("USA CO₂ Emissions and Disasters (1980–2014)", fontsize=16)
ax.set_xlabel("Emissions (Metric Tonnes)", fontsize=14)
ax.set_ylabel("Number of Disasters", fontsize=14)
ax.tick_params(axis="x", labelsize=12, colors="black")
ax.tick_params(axis="y", labelsize=12, colors="black")


# disaster v emissions scatterplot (Canada)
filtered = all_merged_drop_ca[
    (all_merged_drop_ca['Country'] == "Canada") &
    (all_merged_drop_ca['Year'] >= 1980) &
    (all_merged_drop_ca['Year'] <= 2014)
]
filtered = filtered.drop(columns=['Label'])
wide_Canada = filtered.pivot(index='Year', columns='Indicator', values='Value').reset_index()

# Plot
fig11, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=wide_Canada, x='Emissions', y='Disasters', color="black", s=40, ax=ax)
sns.regplot(data=wide_Canada, x='Emissions', y='Disasters', ci=None, scatter=False, color="red", ax=ax)

ax.set_title("Canada CO₂ Emissions and Disasters (1980–2014)", fontsize=16)
ax.set_xlabel("Emissions (Metric Tonnes)", fontsize=14)
ax.set_ylabel("Number of Disasters", fontsize=14)
ax.tick_params(axis="x", labelsize=12, colors="black")
ax.tick_params(axis="y", labelsize=12, colors="black")



st.header("USA and Canada Emissions v. number of Disasters scatterplots") 
st.write("Scatterplot of Emissions v. number of Disasters for both the US (left) and Canada (right) with line of best fit")
col1, col2 = st.columns(2)

with col1:
    st.subheader("USA CO₂ Emissions vs number of Disasters (1980–2014)")
    st.pyplot(fig10)

with col2:
    st.subheader("Canada CO₂ Emissions vs number of Disasters (1980–2014)")
    st.pyplot(fig11)


st.title("Data Analysis")
col 1, col2 = st.columns(2)
st.header("Mean and Standard Deviations") 
with col1:
    st.subheader("USA")
    st.write("mean emissions US: 5142285.714285715 Mt")
    st.write("standard deviation emissions US: 450549.2446434013 Mt")
    st.write("mean temperature US: 52.872 degrees Celsius")
    st.write("standard deviation temperature US: 0.8906600328507205 degrees Celsius")

with col2:
    st.subheader("Canada")
    st.write("mean emissions Canada: 487200.0 Mt")
    st.write("standard deviation emissions Canada: 51928.00445866203 Mt")
    st.write("mean temperature Canada: 0.8285714285714286 degrees Celsius")
    st.write("standard deviation temperature Canada: 0.8753390699580123 degrees Celsius")

st.title("Conclusion")
st.write("My results show that the global CO2 emission rates have drastically increased over time. This can be seen very clearly from my graph Country CO2 emissions per year where there is a drastic upward increase by many countries starting as early as 1900 for some countries (USA). For the US in particular, although we are not the most CO2 emitting nation, we are definetly amongst the top few. This can be clearly seen with the Top 10 CO2 producing Countries graph which the US is most definetly on the leaderboard of. Similar statements can also be made for Canada, because like the US: it is not the most CO2 emitting nation, it is most definetly amongst the top few. Which can also be clearly seen with the Top 10 CO2 producing Countries graph which Canada is also most definetly on the leaderboard of.

I would say that CO2 emissions definetly do have some sort of a correlation to the increasing trend in global temperatures. This is most evidently seen by  both the USA and Canada CO2 emissions and Temperature Scatterplot. The correlation coefficient for the graphs are 0.471 and 0.489 respectively, which indicates a moderately positive correlation between the two variables. As with its association with natural disaster rates, there also seems to be a slight positive correlation as in the plot Canada CO2 emissions and Disasters Scatterplot the correlation coefficient between the two variables is 0.447. However, the correlation between the number of disasters and CO2 emissions for the US is less strongly related as the correlation coefficient is only 0.373.")



