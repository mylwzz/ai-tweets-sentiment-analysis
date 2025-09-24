# src/analyze_local_csv.py
import os, glob
import pandas as pd
from pathlib import Path

from src.cleanvader import preprocess_dataframe

OUT_DIR = Path("outputs"); FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_local_csvs(data_dir="data"):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError("No CSVs in ./data. Put your CSVs there.")

    frames = []
    for f in files:
        df = pd.read_csv(f)

        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

        if "text" not in df.columns:
            raise KeyError(f"{os.path.basename(f)} is missing a 'text' column.")

        if "tweet_created" not in df.columns:
            if "created_at" in df.columns:
                df = df.rename(columns={"created_at": "tweet_created"})
            else:
                raise KeyError(
                    f"{os.path.basename(f)} has no 'tweet_created' or 'created_at' column."
                )

        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    return merged

def main():
    df = load_local_csvs()


    data = preprocess_dataframe(df, text_col="text", time_col="tweet_created")
    data = data.dropna(subset=["datetime"])

    OUT_DIR.mkdir(exist_ok=True)
    data.to_csv(OUT_DIR / "local_clean.csv", index=False)

    # viz
    import matplotlib.pyplot as plt, seaborn as sns
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7,5))
    ax = sns.countplot(data=data, x="vader_sentiment", order=["negative","neutral","positive"])
    ax.set_title("VADER Sentiment (Local CSVs)")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "local_vader_counts.png", dpi=200)
    plt.close()

    print("✓ Wrote outputs/local_clean.csv")
    print("✓ Wrote outputs/figures/local_vader_counts.png")

if __name__ == "__main__":
    main()
