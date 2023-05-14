import glob
from pathlib import Path

import pandas as pd
import plotly.express as px

pd.read_csv(files[0])

dat = pd.read_feather("dat/preprocessed/joint.feather")

dat.groupby(["participant_id", "trial_id"]).index.first().reset_index().drop(
    columns="index"
).groupby("participant_id").trial_id.count()

dat.groupby(["participant_id"]).index.count()

files = glob.glob("dat/FaceWord/logfiles/*.csv")
df = []
for file in files:
    subject = Path(file).stem.split("_")[0]
    try:
        session_df = pd.read_csv(file)
        session_df["subject"] = subject
        df.append(session_df)
    except Exception:
        print(f"{file} faulty!")
df = pd.concat(df)

len(df.groupby("subject").gender.first())

df.groupby("subject").age.first().std()

df.groupby("subject").session.value_counts().reset_index()

df.info()

df.duration_measured_img

df.word_label

px.histogram(df.offset_word - df.onset_word)

px.histogram(df.offset_img - df.onset_img)

px.histogram(df.onset_img - df.offset_word, color=df.subject)

px.histogram(
    df.rt, template="simple_white", height=400, width=1600
).update_layout(xaxis_title="", yaxis_title="")

px.histogram(df.onset_word)
