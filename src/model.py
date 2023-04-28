import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

data = pd.read_feather("dat/preprocessed/joint.feather")
data = data.drop(columns="index")
data_backup = data

data.groupby("participant_id").trial_id.count()

data.condition.unique()


data.participant_id.unique()

data = data[data["elapsed"] != 0]
data = data[data["participant_id"] == "group3"]

data

dummy_condition = pd.get_dummies(data.condition, dtype=int)
sensors = data.columns[:30]
coords = {
    # "conditions": data.condition.unique(),
    "responses": data.response.unique(),
    "participants": data.participant_id.unique(),
    "sensors": sensors,
    "conditions": dummy_condition.columns,
}
with pm.Model(coords=coords) as model:
    condition = pm.MutableData("condition", dummy_condition)
    elapsed_time = pm.MutableData("t", data.elapsed)
    drift_intercept = pm.Normal("drift_intercept", 0, 1, dims=("sensors"))
    dispersion_intercept = pm.HalfCauchy(
        "dispersion_intercept", 1, dims=("sensors")
    )
    condition_drift_effect = pm.Normal(
        "condition_drift_effect", 0, 1, dims=("conditions", "sensors")
    )
    # condition_dispersion_effect = pm.Normal(
    #     "condition_dispersion_effect", 0, 1, dims=("conditions")
    # )
    drift = drift_intercept + pt.dot(condition, condition_drift_effect)
    dispersion = dispersion_intercept
    # + pt.sum(
    #     condition * condition_dispersion_effect
    # )
    mu_sensors = (elapsed_time * drift.T).T
    dispersion_broadcast = pt.broadcast_to(
        dispersion, (elapsed_time.shape[0], dispersion.shape[0])
    ).T
    sd_sensors = pt.sqrt(pt.sqr(dispersion_broadcast) * elapsed_time).T
    sensor_data = pm.Normal(
        "sensor_data", mu_sensors, sd_sensors, observed=data[sensors]
    )

model.debug(verbose=True)


with model:
    var_fit = pm.fit(n=25000, method="advi")
    idata = var_fit.sample()


az.plot_forest(idata, var_names="condition_drift_effect")
plt.show()
