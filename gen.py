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

    df = pd.read_csv( f"{VOLUME_DIR}/{DATASET_FILENAME}", usecols=[IN_COL, OUT_COL])

    # Grouping tracks by artists
    gd = df.groupby(OUT_COL)[IN_COL].agg(list).reset_index()

    itemSetList = list(gd[IN_COL])

    result = fpgrowth(itemSetList, minSupRatio=0.01, minConf=0.2)

    if result == None:
        return

    freqItemSet, rules = result

    with open(f"{VOLUME_DIR}/{OUTPUT_FILENAME}", "wb") as f:
        pickle.dump(rules, f)


if __name__ == "__main__":
    main()
