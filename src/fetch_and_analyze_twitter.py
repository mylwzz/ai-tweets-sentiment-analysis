# src/fetch_and_analyze_twitter.py
import os
import argparse
import pandas as pd
from pathlib import Path
import tweepy

from src.cleanvader import preprocess_dataframe

OUT_DIR = Path("outputs"); FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)

def get_client():
    bearer = os.getenv("BEARER_TOKEN")
    if not bearer:
        raise RuntimeError("Set BEARER_TOKEN (e.g., via .env and `export $(grep -v '^#' .env | xargs)`)")
    return tweepy.Client(bearer_token=bearer, return_type=dict, wait_on_rate_limit=True)

def search_tweets(client, query, max_tweets=100, page_size=100):
    all_tweets, next_token = [], None
    while len(all_tweets) < max_tweets:
        resp = client.search_recent_tweets(
            query=query,
            tweet_fields=["id","text","author_id","created_at","public_metrics","lang"],
            max_results=min(page_size, max_tweets - len(all_tweets)),
            next_token=next_token
        )
        data = resp.get("data", [])
        all_tweets.extend(data)
        next_token = resp.get("meta", {}).get("next_token")
        if not next_token: break
    return pd.DataFrame(all_tweets)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default='(DeepSeek OR #DeepSeek) -is:retweet lang:en')
    ap.add_argument("--max_tweets", type=int, default=100)
    args = ap.parse_args()

    client = get_client()
    raw = search_tweets(client, args.query, args.max_tweets)
    if raw.empty:
        print("No tweets found."); return
    raw.to_csv(OUT_DIR / "api_raw.csv", index=False)

    # normalize for preprocess
    data = raw.rename(columns={"created_at": "tweet_created"})
    data = preprocess_dataframe(data, text_col='text', time_col='tweet_created')
    data.to_csv(OUT_DIR / "api_clean.csv", index=False)

    # viz
    import matplotlib.pyplot as plt, seaborn as sns
    plt.figure(figsize=(7,5))
    ax = sns.countplot(data=data, x='vader_sentiment', order=['negative','neutral','positive'])
    ax.set_title("VADER Sentiment (API)")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()), ha='center', va='bottom')
    plt.tight_layout(); plt.savefig(FIG_DIR / "api_vader_counts.png", dpi=200); plt.close()
    print("Wrote outputs/api_raw.csv, outputs/api_clean.csv and outputs/figures/api_vader_counts.png")

if __name__ == "__main__":
    main()
