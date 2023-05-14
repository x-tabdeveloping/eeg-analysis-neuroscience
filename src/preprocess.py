"""Preprocesses data and saves it to a feather file on disk"""
from pathlib import Path

import pandas as pd

from utils.preprocessing import collect_data, get_trials_data, preprocess_raw

OUT_DIR = "dat/preprocessed"


def main():
    print("Collecting raw data:")
    raws = collect_data(dir_path="dat/FaceWord")
    sessions = []
    print("Accumulating sessions:")
    for name, raw in raws:
        try:
            print(f" - {name}")
            print("   preprocessing...")
            raw = preprocess_raw(raw)
            print("   accumulating trials...")
            session = get_trials_data(raw)
            print("   DONE...")
            session["participant_id"] = name
            session.reset_index().to_feather(
                Path(OUT_DIR).joinpath(f"{name}.feather")
            )
            sessions.append(session)
        except Exception as e:
            print(f"-----------------------------------------")
            print(f"!!!!!!!!!!!!!{name} FAILED!!!!!!!!!!!!!!!")
            print(f"-----------------------------------------")
            print(e)
    print("Concatenating sessions")
    data = pd.concat(sessions)

    out_path = Path(OUT_DIR).joinpath("joint.feather")
    print(f"Saving data to feather file {out_path}")
    data.reset_index().to_feather(out_path)
    print("DONE")


if __name__ == "__main__":
    main()
