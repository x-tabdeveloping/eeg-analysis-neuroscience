import mne

from utils.preprocessing import (
    convert_triggers,
    get_trials_data,
    preprocess_raw,
)

raw = mne.io.read_raw_brainvision("dat/FaceWord/group7.vhdr")
raw = preprocess_raw(raw)

session = get_trials_data(raw)


# Getting events from the data
events, event_id = mne.events_from_annotations(raw)
# Fixing all the stupid annotation things
events, event_id = convert_triggers(events)

# Extracting raw arrays from the events
raw_data, times = raw.get_data(return_times=True)

# Wrangling events into a data frame
events_df = pd.DataFrame(events, columns=["onset", "whatever", "event_id"])
# Mapping event names onto ID's
events_df["event"] = events_df.event_id.map(
    {i: name for name, i in event_id.items()}
)

events_df.event_id.unique()


# We drop the middle column cause it's useless anyways
events_df = events_df.drop(columns="whatever")
events_df = events_df.dropna()
# Split the annotations onto event and condition
events_df[["event", "condition"]] = events_df.event.str.split(
    "/", n=1
).tolist()

len(events_df)
