import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pymc as pm
import pytensor.tensor as pt
from pymc.sampling_jax import sample_numpyro_nuts

from utils.components import (
    components_barplot,
    components_heatmap,
    extract_components_ica,
    name_components,
)
from utils.models import diffusion_model, random_walk_model
from utils.preprocessing import map_polarity_congruency

# Loading data from disk
data = pd.read_feather("../dat/preprocessed/joint.feather")
# Dropping index so that the first 30 columns are sensor data
data = data.drop(columns="index")
data = data[~data.participant_id.isin(["group3"])]
data = map_polarity_congruency(data)
# data = data[data.participant_id == "group6"]

# Removing data points where the elapsed time is 0
# Because dispersion won't work if we leave 'em in there.
data = data[data["elapsed"] != 0]
# Removing trials where the response time is longer than a second.
data = data[np.abs(data.rt) < 1.0]

n_obs = len(data.index)

n_components = 8
fast_ica, reduced, prev_reduced = extract_components_ica(
    data, n_components=n_components
)

components_barplot(fast_ica).show()

component_names = name_components(fast_ica)

component_names

df = pd.DataFrame(reduced, columns=component_names)
df["trial_name"] = data.trial_name
df["elapsed"] = data.elapsed
trials = df["trial_name"].unique()[:10]
df = df[df.trial_name.isin(trials)]
df = df.melt(id_vars=["trial_name", "elapsed"], var_name="component")
px.line(
    df,
    x="elapsed",
    y="value",
    color="component",
    facet_col="trial_name",
    facet_col_wrap=4,
)


t0 = 2
t = t0 + np.arange(100) / 100
drift = 0.3
dispersion = 0.4
y = np.random.normal(drift * t, t * dispersion**2)
px.line(x=t, y=y)

model = diffusion_model(data, reduced, component_names=component_names)

model = random_walk_model(
    data, reduced, prev_reduced, component_names=component_names
)

with model:
    idata = sample_numpyro_nuts()  # pm.sample()

az.plot_forest(idata, var_names="congruency_drift_effect", filter_vars="like")
plt.tight_layout()
plt.show()

with model:
    idata.extend(pm.sample_posterior_predictive(idata))
    az.plot_ppc(idata)
    plt.tight_layout()
    plt.show()


summary = az.summary(
    idata,
    var_names=["effect"],
    filter_vars="like",
)
significant_effects = summary[
    (summary["hdi_3%"] > 0) | (summary["hdi_97%"] < 0)
].index

summary.loc[significant_effects]

az.plot_forest(idata)
plt.tight_layout()
plt.show()
