import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
from pymc.sampling.jax import sample_blackjax_nuts, sample_numpyro_nuts
from scipy.stats import zscore
from sklearn.decomposition import FactorAnalysis, TruncatedSVD
from tqdm import trange

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
# data = data[data.participant_id == "group6"]
data = data[data["elapsed"] != 0]
data = data[np.abs(data.rt) < 1.0]
data["trial_name"] = (
    data["participant_id"] + "/" + data["trial_id"].astype(str)
)
trial_codes, trial_uniques = pd.factorize(data["trial_id"])
data["trial_id"] = trial_codes

prev_data = data[[column for column in data.columns if "prev_" in column]]
prev_data.columns = [
    column.removeprefix("prev_") for column in prev_data.columns
]
prev_data = prev_data[sensors]

svd = TruncatedSVD(n_components=5)
reduced_sensors = svd.fit_transform(data[sensors])

# Topic naming
top_n = 5
components = svd.components_
vocab = svd.feature_names_in_
highest = np.argpartition(-components, top_n)[:, :top_n]
top_words = vocab[highest]
topic_names = []
for i_topic, words in enumerate(top_words):
    name = "_".join(words)
    topic_names.append(f"{i_topic}_{name}")
topic_names = np.array(topic_names)
n_obs, n_components = reduced_sensors.shape

topic_names


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
    elapsed_time = pm.MutableData("t", data.elapsed)
    elapsed_time = pt.broadcast_to(elapsed_time, (n_sensors, n_obs)).T
    # Intercepts
    drift = pm.MvNormal(
        "drift",
        np.zeros(n_sensors),
        np.eye(n_sensors),
        dims=("conditions", "sensors"),
    )
    # drift = pm.Normal(
    #     "drift",
    #     np.zeros(n_sensors),
    #     np.ones(n_sensors),
    #     dims=("conditions", "sensors"),
    # )
    dispersion = pm.HalfCauchy("dispersion", 1, dims=("conditions", "sensors"))
    drift_magnitude = pm.Deterministic(
        "drift_magnitude", drift.norm(L=2, axis=-1), dims="conditions"
    )
    dispersion_magnitude = pm.Deterministic(
        "dispersion_magnitude",
        dispersion.norm(L=2, axis=-1),
        dims="conditions",
    )
    outcome = pm.Normal(
        "outcome",
        elapsed_time * drift[condition],
        pt.sqrt(pt.sqr(dispersion[condition]) * elapsed_time),
        observed=data[sensors],
    )
model.debug(verbose=True)

with model:
    var_fit = pm.fit(n=6000, method="advi")
    idata = var_fit.sample()

summary = az.summary(
    idata,
    var_names=["dispersion", "drift", "epsilon", "intercept"],
    filter_vars="like",
)

summary[(summary["hdi_3%"] > 0) | (summary["hdi_97%"] < 0)]

az.plot_forest(idata, var_names="drift_magnitude", filter_vars="like")
plt.tight_layout()
plt.show()

with model:
    idata.extend(pm.sample_posterior_predictive(idata))
    az.plot_ppc(idata)
    plt.tight_layout()
    plt.show()


data

px.line(data[data.trial_id == 8], x="elapsed", y="Fp1")
