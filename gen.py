import pandas as pd
from fpgrowth_py import fpgrowth

import pickle

# Dummy Test Files
DATASETS_FOLDER = "."
TRAIN_DATA_FILENAME = "food.csv"

# Data file constants
"""
DATASETS_FOLDER = "/home/datasets"
TRAIN_DATA_FILENAME = "2023_spotify_ds1.csv"
UPDATE_DATA_FILENAME = "2023_spotify_ds2.csv"
SONGS_FILENAME = "2023_spotify_songs.csv"
"""

# Persistent Volume Access
OUTPUT_FOLDER = "/home/bernardoborges/project2-pv"
OUTPUT_FILENAME = "model.pkl"


def main():
    # Reading input file with relevant columns
    IN_COL = "track_name"
    OUT_COL = "pid"

    df = pd.read_csv(
        f"{DATASETS_FOLDER}/{TRAIN_DATA_FILENAME}", usecols=[IN_COL, OUT_COL]
    )

    # Grouping tracks by artists
    gd = df.groupby(OUT_COL)[IN_COL].agg(list).reset_index()

    itemSetList = list(gd[IN_COL])

    result = fpgrowth(itemSetList, minSupRatio=0.1, minConf=0.5)

    if result == None:
        return

    freqItemSet, rules = result

    with open(f"{OUTPUT_FOLDER}/{OUTPUT_FILENAME}", "wb") as f:
        pickle.dump(rules, f)


if __name__ == "__main__":
    main()
