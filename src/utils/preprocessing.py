"""Module with utilities for preprocessing EEG data for analysis."""
from glob import glob
from pathlib import Path

import mne
import numpy as np
import pandas as pd


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


def get_trials_data(data: mne.io.Raw) -> pd.DataFrame:
    """Extracts trial data from the raw datastructure.
    By that I mean that it takes all sensor data from the image shown
    to the response with timestamps as well as elapsed time
    from trial onset (image onset)."""
    # Getting events from the data
    events, event_id = mne.events_from_annotations(data)
    # Fixing all the stupid annotation things
    events, event_id = convert_triggers(events)

    # Extracting raw arrays from the events
    raw_data, times = data.get_data(return_times=True)

    # Wrangling events into a data frame
    events_df = pd.DataFrame(events, columns=["onset", "whatever", "event_id"])
    # Mapping event names onto ID's
    events_df["event"] = events_df.event_id.map(
        {i: name for name, i in event_id.items()}
    )
    # We drop the middle column cause it's useless anyways
    events_df = events_df.drop(columns="whatever")
    events_df = events_df.dropna()
    # Split the annotations onto event and condition
    events_df[["event", "condition"]] = events_df.event.str.split(
        "/", n=1
    ).tolist()
    # Filtering for events that are either the image onset or a response
    events_df = events_df[
        events_df.event.isin(["Image", "Correct", "Incorrect"])
    ]
    # Lagging the data, so we have the previous event's data there too
    # I do this so that I can get the onset of a trial (image onset)
    # on the same row as the trial end (response)
    events_df["prev_event"] = events_df.event.shift(periods=1)
    events_df["prev_onset"] = events_df.onset.shift(periods=1)
    events_df["prev_condition"] = events_df.condition.shift(periods=1)
    # Filter for responses
    events_df = events_df[events_df.event.isin(["Correct", "Incorrect"])]
    # Renaming event to response,
    # Cause only correct and incorrect events are left
    events_df["response"] = events_df["event"]
    events_df = events_df.dropna()
    events_df = events_df.astype({"onset": int, "prev_onset": int})
    # Adding onset and end times from the raw data
    events_df["onset_time"] = times[events_df["prev_onset"]]
    events_df["end_time"] = times[events_df["onset"]]

    # Collecting data from different trials to a list of
    # dataframes>
    sensors = data.info.ch_names
    trials = []
    for trial_index, event in events_df.iterrows():
        trial_sensor_data = raw_data[:, event.prev_onset : event.onset].T
        # We want all sensors to start from 0, since we're
        # modelling the motion of the sensors, not their initial state
        trial_sensor_data = trial_sensor_data - trial_sensor_data[0, :]
        trial = pd.DataFrame(trial_sensor_data, columns=sensors)
        for sensor in sensors:
            trial[f"prev_{sensor}"] = trial[sensor].shift(1)
        trial["trial_id"] = trial_index
        trial["time"] = times[event.prev_onset : event.onset]
        trial["rt"] = event.end_time - event.onset_time
        trial["elapsed"] = trial["time"] - event.onset_time
        trial["condition"] = event.condition
        trial["response"] = event.response
        trials.append(trial)
    # Concatenating them all, so we have a joint one for the whole session
    session = pd.concat(trials)
    return session


def preprocess_raw(raw: mne.io.Raw) -> mne.io.Raw:
    """Preprocesses raw files to be usable in analyses."""
    raw = raw.copy()
    # Excluding the HEOG and VEOG channels, only eeping EEG
    raw.pick_types(
        meg=False, eeg=True, stim=True, eog=False, exclude=["VEOG", "HEOG"]
    )
    # Resampling data for faster processing
    raw.resample(100, npad="auto")
    # Set average reference
    raw.set_eeg_reference(ref_channels="average")
    # high-pass filter the data
    raw = raw.filter(0.1, None)
    # low-pass filter the data
    raw = raw.filter(None, 40)
    return raw


def collect_data(dir_path: str) -> list[tuple[str, mne.io.Raw]]:
    """Collects all raw data objects from a given directory along
    with their names."""
    paths = glob(str(Path(dir_path).joinpath("group*.vhdr")))
    print("Found the following paths:")
    for path in paths:
        print(f" - {path}")
    raws = []
    for path in paths:
        name = Path(path).stem
        raw = mne.io.read_raw_brainvision(path, preload=True)
        raws.append((name, raw))
    return raws


def collect_preprocess_data(dir_path: str) -> pd.DataFrame:
    """Collects all EEG data from the given directory,
    runs preprocessing on the raw data, isolates trials, and wrangles
    the data into a DataFrame."""
    print("Collecting raw data:")
    raws = collect_data(dir_path)
    sessions = []
    print("Accumulating sessions:")
    for name, raw in raws:
        print(f" - {name}")
        print("   preprocessing...")
        raw = preprocess_raw(raw)
        print("   accumulating trials...")
        session = get_trials_data(raw)
        print("   DONE...")
        session["participant_id"] = name
    print("Concatenating sessions")
    return pd.concat(sessions)


def map_polarity_congruency(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    condition_mapping = {
        "wNeg": "Negative/Congruent",
        "wNeu/iPos": "Positive/Neutral",
        "wNeu/iNeg": "Negative/Neutral",
        "Neu/iNeg": "Negative/Neutral",
        "wPos": "Positive/Congruent",
    }
    data["condition"] = data.condition.map(condition_mapping)
    data[["polarity", "congruency"]] = data.condition.str.split("/").tolist()
    data["trial_name"] = (
        data["participant_id"] + "/" + data["trial_id"].astype(str)
    )
    return data
