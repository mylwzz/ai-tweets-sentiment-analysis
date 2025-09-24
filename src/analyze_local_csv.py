# src/analyze_local_csv.py
import os, glob
import pandas as pd
from pathlib import Path
from common import preprocess_dataframe

OUT_DIR = Path("outputs"); FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)

# If your time column isn't 'tweet_created', change it below (e.g. 'created_at')
TEXT_COL = 'text'
TIME_COL = 'tweet_created'  # change to 'created_at' if needed

def load_local_csvs(data_dir="data"):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError("No CSVs in ./data. Put chatgpt_daily_tweets.csv / tweets.csv there.")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def main():
    df = load_local_csvs()
    # adjust TIME_COL if your CSV uses 'created_at'
    data = preprocess_dataframe(df, text_col=TEXT_COL, time_col=TIME_COL)
    data.to_csv(OUT_DIR / "local_clean.csv", index=False)

    # tiny visualization
    import matplotlib.pyplot as plt, seaborn as sns
    plt.figure(figsize=(7,5))
    ax = sns.countplot(data=data, x='vader_sentiment', order=['negative','neutral','positive'])
    ax.set_title("VADER Sentiment (Local CSVs)")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()), ha='center', va='bottom')
    plt.tight_layout(); plt.savefig(FIG_DIR / "local_vader_counts.png", dpi=200); plt.close()
    print("Wrote outputs/local_clean.csv and outputs/figures/local_vader_counts.png")

if __name__ == "__main__":
    main()
