import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


def random_walk_model(
    data: pd.DataFrame,
    reduced: np.ndarray,
    reduced_prev: np.ndarray,
    component_names: list[str],
) -> pm.Model:
    """Creates random walk model from given data.

    Parameters
    ----------
    data: DataFrame
        Trials data.
    reduced: array of shape (n_observations, n_components)
        Data of independent components.
    reduced_prev: array of shape (n_observations, n_components)
        Data of independent components lagging one behind.
    component_names: list of str
        Names of components.

    Returns
    -------
    Model
        PyMC model.
    """
    n_obs, n_components = reduced_prev.shape
    coords = {
        "responses": data.response.unique(),
        "participants": data.participant_id.unique(),
        "components": component_names,
        "congruency": ["congruent", "incongruent"],
        "positivity": ["positive", "negative"],
    }
    with pm.Model(coords=coords) as model:
        positivity = pm.MutableData(
            "positivity", (data["polarity"] == "Positive").astype(int)
        )
        congruency = pm.MutableData(
            "congruency", (data["congruency"] == "Congruent").astype(int)
        )
        prev_outcome = pm.MutableData("prev", reduced_prev)
        interaction = pm.Deterministic("interaction", positivity * congruency)
        interaction = pt.broadcast_to(interaction, (n_components, n_obs)).T
        positivity = pt.broadcast_to(positivity, (n_components, n_obs)).T
        congruency = pt.broadcast_to(congruency, (n_components, n_obs)).T
        autocorrelation = pm.Normal(
            "autocorrelation", 0, 1, dims=("components")
        )
        effect_congruency_autocorrelation = pm.Normal(
            "effect_congruency_autocorrelation", 0, 1, dims=("components")
        )
        effect_congruency_drift = pm.Normal(
            "effect_congruency_drift", 0, 1, dims=("components")
        )
        effect_positivity_autocorrelation = pm.Normal(
            "effect_positivity_autocorrelation", 0, 1, dims=("components")
        )
        effect_positivity_drift = pm.Normal(
            "effect_positivity_drift", 0, 1, dims=("components")
        )
        effect_interaction_autocorrelation = pm.Normal(
            "effect_interaction_autocorrelation", 0, 1, dims=("components")
        )
        effect_interaction_drift = pm.Normal(
            "effect_interaction_drift", 0, 1, dims=("components")
        )
        drift = pm.Normal("drift", 0, 1, dims=("components"))
        epsilon = pm.HalfCauchy("epsilon", 1, dims=("components",))
        _autocorr = (
            effect_congruency_autocorrelation * congruency
            + effect_positivity_autocorrelation * positivity
            + effect_interaction_autocorrelation * interaction
            + autocorrelation
        )
        _drift = (
            effect_congruency_drift * congruency
            + effect_positivity_drift * positivity
            + effect_interaction_drift * interaction
            + drift
        )
        mean_outcome = _autocorr * prev_outcome + _drift
        outcome = pm.Normal("outcome", mean_outcome, epsilon, observed=reduced)
    model.debug(verbose=True)
    return model


def diffusion_model(
    data: pd.DataFrame,
    reduced: np.ndarray,
    component_names: list[str],
) -> pm.Model:
    """Creates diffusion model given the data.

    Parameters
    ----------
    data: DataFrame
        Trials data.
    reduced: array of shape (n_observations, n_components)
        Data of independent components.
    component_names: list of str
        Names of components.

    Returns
    -------
    Model
        PyMC model.
    """
    n_obs, n_components = reduced.shape
    coords = {
        # "conditions": data.condition.unique(),
        "responses": data.response.unique(),
        "participants": data.participant_id.unique(),
        "components": [f"Component {i}" for i in range(n_components)],
    }
    with pm.Model(coords=coords) as model:
        positivity = pm.MutableData(
            "positivity", (data["polarity"] == "Positive").astype(int)
        )
        congruency = pm.MutableData(
            "congruency", (data["congruency"] == "Congruent").astype(int)
        )
        interaction = pm.Deterministic("interaction", positivity * congruency)
        interaction = pt.broadcast_to(interaction, (n_components, n_obs)).T
        positivity = pt.broadcast_to(positivity, (n_components, n_obs)).T
        congruency = pt.broadcast_to(congruency, (n_components, n_obs)).T
        elapsed_time = pm.MutableData("t", data.elapsed)
        elapsed_time = pt.broadcast_to(elapsed_time, (n_components, n_obs)).T
        # Intercepts
        t0 = pm.Normal("t0", 0, 1)
        # epsilon = pm.HalfCauchy("epsilon", 1, dims="components")
        drift_intercept = pm.Normal("drift_intercept", 0, 1, dims="components")
        dispersion_intercept = pm.HalfCauchy(
            "dispersion_intercpt", 1, dims="components"
        )
        # Effects
        # congruency
        congruency_drift_effect = pm.Normal(
            "congruency_drift_effect", 0, 1, dims="components"
        )
        congruency_dispersion_effect = pm.Normal(
            "congruency_dispersion_effect", 0, 1, dims="components"
        )
        # positivity
        positivity_drift_effect = pm.Normal(
            "positivity_drift_effect", 0, 1, dims="components"
        )
        positivity_dispersion_effect = pm.Normal(
            "positivity_dispersion_effect", 0, 1, dims="components"
        )
        # interaction
        interaction_drift_effect = pm.Normal(
            "interaction_drift_effect", 0, 1, dims="components"
        )
        interaction_dispersion_effect = pm.Normal(
            "interaction_dispersion_effect", 0, 1, dims="components"
        )
        # Drift and dispersion
        drift = (
            drift_intercept
            + positivity * positivity_drift_effect
            + congruency * congruency_drift_effect
            + interaction * interaction_drift_effect
        )
        dispersion = (
            dispersion_intercept
            + positivity * positivity_dispersion_effect
            + congruency * congruency_dispersion_effect
            + interaction * interaction_dispersion_effect
        )
        t = elapsed_time + t0
        mu_sensors = t * drift  # + intercept
        sd_sensors = pt.sqrt(pt.sqr(dispersion) * t)  # + epsilon
        outcome = pm.Normal(
            "outcome",
            mu_sensors,
            sd_sensors,
            observed=reduced,
        )
    model.debug(verbose=True)
    return model
