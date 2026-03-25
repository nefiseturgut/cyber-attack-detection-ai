import pandas as pd
from pathlib import Path

DATA_DIR = Path("datasets/CICIDS2018")
CSV_FILE = "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"

def main():
    csv_path = DATA_DIR / CSV_FILE
    print(f"Loading file: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    print("\nDataset shape:")
    print(df.shape)

    print("\nLabel distribution:")
    print(df["Label"].value_counts().head(15))

if __name__ == "__main__":
    main()
