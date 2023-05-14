import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
from pymc.sampling_jax import sample_numpyro_nuts
from scipy.stats import zscore
from sklearn.decomposition import FactorAnalysis, FastICA, TruncatedSVD
from tqdm import trange

n_sensors = 30

data = pd.read_feather("dat/preprocessed/joint.feather")
data = data.drop(columns="index")
condition_mapping = {
    "wNeg": "Negative/Congruent",
    "wNeu/iPos": "Positive/Neutral",
    "wNeu/iNeg": "Negative/Neutral",
    "Neu/iNeg": "Negative/Neutral",
    "wPos": "Positive/Congruent",
}
data["condition"] = data.condition.map(condition_mapping)
data[["polarity", "congruency"]] = data.condition.str.split("/").tolist()
data = data[~data.participant_id.isin(["group7", "group3"])]
data = data[data.participant_id == "group6"]
data = data[data["elapsed"] != 0]
data = data[np.abs(data.rt) < 1.0]
data["trial_name"] = (
    data["participant_id"] + "/" + data["trial_id"].astype(str)
)
trial_codes, trial_uniques = pd.factorize(data["trial_id"])
data["trial_id"] = trial_codes
sensors = data.columns[:n_sensors]
n_obs = len(data.index)

prev_data = data[[column for column in data.columns if "prev_" in column]]
prev_data.columns = [
    column.removeprefix("prev_") for column in prev_data.columns
]
prev_data = prev_data[sensors]


n_components = 8
dim_red = FastICA(n_components=n_components)
reduced_sensors = dim_red.fit_transform(data[sensors])
reduced_prev = dim_red.transform(prev_data)
component_names = [f"Component {i}" for i in range(n_components)]
components = pd.DataFrame(
    dim_red.components_, index=component_names, columns=sensors
)

px.imshow(components)

px.bar(
    components.reset_index(names="component").melt(id_vars="component"),
    facet_col="component",
    color="variable",
    x="value",
)

px.bar(components.T)


sensors = data.columns[:30]
condition_codes, condition_uniques = pd.factorize(data.condition)
n_obs = len(data.index)
n_sensors = 30
coords = {
    "conditions": condition_uniques,
    "responses": data.response.unique(),
    "participants": data.participant_id.unique(),
    "sensors": sensors,
}
with pm.Model(coords=coords) as model:
    condition = pm.MutableData("condition", condition_codes)
    prev_outcome = pm.MutableData("prev", prev_data)
    autocorrelation = pm.MvNormal(
        "autocorrelation",
        np.zeros(n_sensors),
        np.eye(n_sensors),
        dims=("conditions", "sensors"),
    )
    drift = pm.MvNormal(
        "drift",
        np.zeros(n_sensors),
        np.eye(n_sensors),
        dims=("conditions", "sensors"),
    )
    drift_magnitude = pm.Deterministic(
        "drift_magnitude", drift.norm(L=2, axis=-1), dims="conditions"
    )
    autocorr_magnitude = pm.Deterministic(
        "autocorr_magnitude",
        autocorrelation.norm(L=2, axis=-1),
        dims="conditions",
    )
    epsilon = pm.HalfCauchy("epsilon", 1)
    mean_outcome = autocorrelation[condition] * prev_outcome + drift[condition]
    outcome = pm.Normal(
        "outcome", mean_outcome, epsilon, observed=data[sensors]
    )
model.debug(verbose=True)

with model:
    idata = pm.sample()  # sample_numpyro_nuts()
    # var_fit = pm.fit(n=100000, method="advi")
    # idata = var_fit.sample()


az.plot_forest(idata, var_names="_drift", filter_vars="like")
plt.tight_layout()
plt.show()

summary = az.summary(
    idata,
    var_names=["effect"],
    filter_vars="like",
)
summary[(summary["hdi_3%"] > 0) | (summary["hdi_97%"] < 0)]


with model:
    idata.extend(pm.sample_posterior_predictive(idata))
    az.plot_ppc(idata)
    plt.tight_layout()
    plt.show()
