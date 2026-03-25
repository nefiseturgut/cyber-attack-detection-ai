import pandas as pd
from pathlib import Path

TRAIN = Path("datasets/UNSW_NB15/UNSW_NB15_training-set.csv")
TEST  = Path("datasets/UNSW_NB15/UNSW_NB15_testing-set.csv")

def main():
    df_tr = pd.read_csv(TRAIN, low_memory=False)
    df_te = pd.read_csv(TEST, low_memory=False)

    print("Train shape:", df_tr.shape)
    print("Test  shape:", df_te.shape)

    if "label" in df_tr.columns:
        print("\nBinary label distribution (train):")
        print(df_tr["label"].value_counts())

    if "attack_cat" in df_tr.columns:
        print("\nAttack category distribution (train):")
        print(df_tr["attack_cat"].value_counts().head(15))

if __name__ == "__main__":
    main()
