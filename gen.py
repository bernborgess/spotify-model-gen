import pandas as pd
from fpgrowth_py import fpgrowth
import os
import pickle

# Data file constants
VOLUME_DIR = os.environ.get("VOLUME_DIR")
DATASET_FILENAME = os.environ.get("DATASET_FILENAME")
OUTPUT_FILENAME = "model.pkl"


def main():
    # Testing the container
    print(f"Take the dataset from {VOLUME_DIR}/{DATASET_FILENAME}")
    print(f"Store it in {VOLUME_DIR}/{OUTPUT_FILENAME}")

    # Reading input file with relevant columns
    IN_COL = "track_name"
    OUT_COL = "pid"
    df = pd.read_csv(f"{VOLUME_DIR}/{DATASET_FILENAME}", usecols=[IN_COL, OUT_COL])

    # Map unique pids to indices
    unique_pids = df[OUT_COL].unique()
    pid_to_index = {pid: idx for idx, pid in enumerate(unique_pids)}

    # Create a list of sets
    sets_list = [set() for _ in unique_pids]

    # Insert track_names into sets
    for _, row in df.iterrows():
        pid = row[OUT_COL]
        track_name = row[IN_COL]
        sets_list[pid_to_index[pid]].add(track_name)

    # Transform the list of sets into a list of lists
    rules = [list(s) for s in sets_list]

    with open(f"{VOLUME_DIR}/{OUTPUT_FILENAME}", "wb") as f:
        pickle.dump(rules, f)


if __name__ == "__main__":
    main()
