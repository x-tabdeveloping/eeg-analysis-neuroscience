import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import FastICA

N_SENSORS = 30


def extract_components_ica(
    data: pd.DataFrame, n_components: int
) -> tuple[FastICA, np.ndarray, np.ndarray]:
    """Extracts independent components from the sensor data.

    Parameters
    ----------
    data: DataFrame
        Frame containing the sensor data.
    n_components: int
        Number of components to extract.

    Returns
    -------
    fast_ica: FastICA
        Fitted instance of FastICA.
    reduced: ndarray of shape (n_obs, n_components)
        Reduced sensor data.
    reduced_prev: ndarray of shape (n_obs, n_components)
        Reduced data of the i-1th observations.
    """
    sensors = data.columns[:N_SENSORS]
    prev_data = data[[column for column in data.columns if "prev_" in column]]
    prev_data.columns = [
        column.removeprefix("prev_") for column in prev_data.columns
    ]
    prev_data = prev_data[sensors]
    dim_red = FastICA(n_components=n_components)
    reduced_sensors = dim_red.fit_transform(data[sensors])
    reduced_prev = dim_red.transform(prev_data)
    return dim_red, reduced_sensors, reduced_prev


def components_heatmap(fast_ica: FastICA) -> go.Figure:
    """Plots independent components as heatmap.

    Parameters
    ----------
    fast_ica: FastICA
        Fitted instance of FastICA.

    Returns
    -------
    Figure
        Heatmap of components and sensors.
    """
    n_components, _ = fast_ica.components_.shape
    component_names = [f"Component {i}" for i in range(n_components)]
    components = pd.DataFrame(
        fast_ica.components_,
        index=component_names,
        columns=fast_ica.feature_names_in_,
    )
    return px.imshow(components)


def components_barplot(fast_ica: FastICA) -> go.Figure:
    """Plots independent components as barplots of sensor importance.

    Parameters
    ----------
    fast_ica: FastICA
        Fitted instance of FastICA.

    Returns
    -------
    Figure
        Barplot of components and sensors.
    """
    n_components, _ = fast_ica.components_.shape
    component_names = [f"Component {i}" for i in range(n_components)]
    components = pd.DataFrame(
        fast_ica.components_,
        index=component_names,
        columns=fast_ica.feature_names_in_,
    )
    return px.bar(
        components.reset_index(names="component").melt(id_vars="component"),
        facet_col="component",
        color="variable",
        x="value",
    )


def name_components(fast_ica: FastICA) -> list[str]:
    top_n = 5
    components = fast_ica.components_
    vocab = fast_ica.feature_names_in_
    highest = np.argpartition(-components, top_n)[:, :top_n]
    top_words = vocab[highest]
    component_names = []
    for i_topic, words in enumerate(top_words):
        name = "_".join(words)
        component_names.append(f"{i_topic}_{name}")
    return component_names
