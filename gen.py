import pandas as pd
from fpgrowth_py import fpgrowth

import pickle

# Data file constants
DATASETS_FOLDER = "/home/datasets"
TRAIN_DATA_FILENAME = "2023_spotify_ds1.csv"
UPDATE_DATA_FILENAME = "2023_spotify_ds2.csv"
SONGS_FILENAME = "2023_spotify_songs.csv"

# Persistent Volume Access
PV_FOLDER = "/home/bernardoborges/project2-pv"

def main():
    # Reading input file with relevant columns
    df = pd.read_csv(
        f"{DATASETS_FOLDER}/{TRAIN_DATA_FILENAME}", usecols=["pid", "artist_name"]
    )

    # Grouping tracks by artists
    gd = df.groupby("pid")["artist_name"].agg(list).reset_index()

    itemSetList = list(gd["artist_name"])

    print(itemSetList)

    result = fpgrowth(itemSetList, minSupRatio=0.05, minConf=0.5)

    if result == None:
        return

    freqItemSet, rules = result

    print(rules)

    with open(f"{PV_FOLDER}/model.pkl", "wb") as f:
        pickle.dump(rules, f)


if __name__ == "__main__":
    main()
