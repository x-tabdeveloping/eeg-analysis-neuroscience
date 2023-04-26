import glob

import mne
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mne.preprocessing import (
    ICA,
    corrmap,
    create_ecg_epochs,
    create_eog_epochs,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = mne.io.read_raw_brainvision("FaceWord/group1.vhdr", preload=True)
data.pick_types(
    meg=False, eeg=True, stim=True, eog=False, exclude=["VEOG", "HEOG"]
)

montages = mne.channels.get_builtin_montages()

for montage_name in montages:
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        data.set_montage(montage, verbose=False)
        print(f"Montage work: {montage_name}")
    except Exception:
        print(f"Montage don't work: {montage_name}")


data.set_montage(montage, verbose=False)

eog_evoked = create_eog_epochs(data, ch_name=["HEOG", "VEOG"]).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()


def convert_triggers(events: np.ndarray) -> tuple[np.ndarray, dict]:
    """Function to convert triggers to failed and successful inhibition."""
    events_tmp = events.copy()
    # What the hell is this, seriously what the steaming
    # pile of shit
    for idx, line in enumerate(events_tmp):
        if any(line[2] == [101, 102, 111, 112]):
            events_tmp[idx - 1][2] = events_tmp[idx - 1][2] + 5
            events_tmp[idx - 2][2] = events_tmp[idx - 2][2] + 5
            events_tmp[idx - 3][2] = events_tmp[idx - 3][2] + 5
    event_id = {
        "Word/wPos": 16,
        "Wait/wPos": 36,
        "Image/wPos": 26,
        "Word/wNeg": 17,
        "Wait/wNeg": 37,
        "Image/wNeg": 27,
        "Word/wNeu": 18,
        "Wait/wNeu/iPos": 56,  # positive image
        "Image/wNeu/iPos": 46,  # positive image
        "Wait/wNeu/iNeg": 57,  # negative image
        "Image/wNeu/iNeg": 47,  # negative image
        "Correct/wPos": 101,  # correct response ('b') to positive word + image
        "Correct/wNeg": 102,  # correct response ('y') to negative word + image
        "Correct/wNeu/iPos": 111,  # correct response ('b') to neutral word + positive image
        "Correct/wNeu/iNeg": 112,  # correct response ('y') to neutral word + negative image
        "Incorrect/wPos": 202,  # incorrect response ('y') to positive word + image
        "Incorrect/wNeg": 201,  # incorrect response ('b') to negative word + image
        "Incorrect/wNeu/iPos": 212,  # incorrect response ('y') to neutral word + positive image
        "Incorrect/Neu/iNeg": 211,  # incorrect response ('b') to neutral word + negative image
    }
    return events_tmp, event_id


events, event_id = mne.events_from_annotations(data)
events, event_id = convert_triggers(events)

raw_data, times = data.get_data(return_times=True)

# Wrangling events into a data frame
events_df = pd.DataFrame(events, columns=["onset", "whatever", "event_id"])
events_df["event"] = events_df.event_id.map(
    {i: name for name, i in event_id.items()}
)
events_df = events_df.drop(columns="whatever")
events_df = events_df.dropna()
events_df[["event", "condition"]] = events_df.event.str.split(
    "/", n=1
).tolist()
events_df = events_df[events_df.event.isin(["Image", "Correct", "Incorrect"])]
events_df["prev_event"] = events_df.event.shift(periods=1)
events_df["prev_onset"] = events_df.onset.shift(periods=1)
events_df["prev_condition"] = events_df.condition.shift(periods=1)
events_df = events_df[events_df.event.isin(["Correct", "Incorrect"])]
events_df["response"] = events_df["event"]
events_df = events_df.dropna()
events_df = events_df.astype({"onset": int, "prev_onset": int})
events_df["onset_time"] = times[events_df["prev_onset"]]
events_df["end_time"] = times[events_df["onset"]]

events_df

raw_data.shape

trial_sensor_data = []
trial_times = []
trial_event = []

sensors = data.info.ch_names
trials = []
for trial_index, event in events_df.iterrows():
    trial_sensor_data = raw_data[:, event.prev_onset : event.onset].T
    # Substracting first line because we're only interested in movement
    trial_sensor_data = trial_sensor_data - trial_sensor_data[0, :]
    trial = pd.DataFrame(trial_sensor_data, columns=sensors)
    trial["trial_id"] = trial_index
    trial["time"] = times[event.prev_onset : event.onset]
    trial["elapsed"] = trial["time"] - event.onset_time
    trial["condition"] = event.condition
    trial["response"] = event.response
    trials.append(trial)

px.line(trials[0].Pz)

run = pd.concat(trials)

px.histogram(events_df.response)

run.info()

epochs = mne.Epochs(data.resample(250), events, event_id)

vars(epochs)

epoch_data = epochs.get_data()

epochs.get_annotations_per_epoch()


epochs_df = epochs.to_data_frame()


annotations = epochs_df.condition.map(lambda s: s.split("/")[1])
sensors = epochs_df.columns[3:]

epochs_df.condition.unique()

event_desc = {
    i_event: event_name
    for event_name, i_event in event_id.items()
    if event_name.startswith("Image")
    or event_name.startswith("Correct")
    or event_name.startswith("Incorrect")
}
annotations = mne.annotations_from_events(
    events, sfreq=0.01, event_desc=event_desc, orig_time=data.info["meas_date"]
)


raw_data.shape
times

annotations.to_data_frame()

trial_raws = data.crop_by_annotations(annotations)

annotations.to_data_frame().onset.astype(int)

data.to_data_frame()

onset = data.annotations.to_data_frame().onset[0]
onset

epochs_df.info()

X = epochs_df[sensors]
x, y = make_pipeline(StandardScaler(), TSNE()).fit_transform(X).T

epochs_df["x"] = x
epochs_df["y"] = y

epochs_df[epochs_df.time == 0]

traces = []
for condition, condition_df in epochs_df.groupby("condition"):
    trace_x = []
    trace_y = []
    for epoch, epoch_df in condition_df.groupby("epoch"):
        # trace_x.append(epoch_df.x.iloc[0])
        # trace_y.append(epoch_df.y.iloc[0])
        # trace_x.append(epoch_df.x[epoch_df.time == 0].iloc[0])
        # trace_y.append(epoch_df.y[epoch_df.time == 0].iloc[0])
        # trace_x.append(epoch_df.x.iloc[-1])
        # trace_y.append(epoch_df.y.iloc[-1])
        trace_x.extend(epoch_df.x)
        trace_y.extend(epoch_df.y)
        trace_x.append(None)
        trace_y.append(None)
    trace = go.Scatter(
        name=condition,
        x=trace_x,
        y=trace_y,
        mode="lines+markers",
        marker=dict(symbol="arrow", size=15, angleref="previous"),
    )
    traces.append(trace)
go.Figure(data=traces)

px.scatter_3d(x=x, y=y, z=z, color=epochs_df.condition)

px.bar(
    epochs_df[epochs_df.condition == "Correct/wPos"],
    x="time",
    y="FC6",
    facet_col="epoch",
)

sensors

px.histogram(
    epochs_df[sensors].melt(value_vars=sensors, var_name="sensor"),
    x="value",
    facet_col="sensor",
    facet_col_wrap=4,
)

data.plot()

data.info.ch_names

montage = mne.channels.make_standard_montage("standard_1020")
data.set_montage(montage, verbose=False)

layout = mne.channels.make_eeg_layout(data.info)
layout.plot()

data.set_eeg_reference(ref_channels="average", ch_type="eeg")
data.plot()

log = pd.read_csv(glob.glob("FaceWord/logfiles/group1_sess_6 *.csv")[0])
log.info()
