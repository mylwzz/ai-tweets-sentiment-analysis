# src/cleanvader.py
import re
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()

MENTION_RE = re.compile(r'@[^\s]+')
HASHTAG_RE = re.compile(r'\B#\S+')
URL_RE = re.compile(r"http\S+")
NONALNUM_RE = re.compile(r'\w+')
SINGLE_CHAR_RE = re.compile(r'\s+[a-zA-Z]\s+')

def clean_text(text: str) -> str:
    t = str(text).lower()
    t = MENTION_RE.sub('', t)
    t = HASHTAG_RE.sub('', t)
    t = URL_RE.sub('', t)
    t = ' '.join(NONALNUM_RE.findall(t))
    t = SINGLE_CHAR_RE.sub(' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def label_sentiment(v: float) -> str:
    if v < -0.05:
        return 'negative'
    if v > 0.35:
        return 'positive'
    return 'neutral'

def preprocess_dataframe(df: pd.DataFrame, text_col: str, time_col: str) -> pd.DataFrame:
    data = df.copy()
    data['original_tweet'] = data[text_col]
    data['datetime'] = pd.to_datetime(data[time_col], errors='coerce').dt.tz_localize(None)
    data['text'] = data[text_col].astype(str).apply(clean_text)
    data['vader_score'] = data['text'].apply(lambda t: sia.polarity_scores(t)['compound'])
    data['vader_sentiment'] = data['vader_score'].apply(label_sentiment)
    return data
