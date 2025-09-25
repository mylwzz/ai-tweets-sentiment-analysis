# ChatGPT & DeepSeek Sentiment Analysis

This project explores how public sentiment toward ChatGPT and DeepSeek unfolds on Twitter/X by combining **historical tweet datasets** with **live-scraped data from the Twitter API**. The pipeline integrates preprocessing, sentiment scoring, clustering, and correlation analyses into a reproducible framework.

## Highlights
- **End-to-End ETL Pipeline**: Ingests CSVs (Kaggle) and live Twitter API data, harmonizes schemas, cleans text (mentions, hashtags, URLs), and extracts linguistic/temporal features.
- **Dual Sentiment Models**: Applied **VADER** and **TextBlob** to compare polarity, validate across models (74.6% overall agreement), and analyze disagreement categories.
- **Semantic Clustering**: Generated embeddings with **SentenceTransformers**, applied **MiniBatchKMeans** and **UMAP** for thematic discovery across 43K+ tweets.
- **Correlation Analysis**: Linked sentiment, user metadata, and engagement metrics; revealed that **informative/neutral tweets drive as much engagement as opinionated ones**.
- **Visualization**: Built clear outputs (heatmaps, confusion matrices, cluster plots, sentiment distributions) to interpret results.
- **Reproducible Setup**: Modular `src/` code, `requirements.txt`, `.env` credential handling, and GitHub-friendly `.gitignore`.

## Outputs
- Clean datasets (`outputs/local_clean.csv`)
- Some quick figures @ (`outputs/figures/`) 
- Please reach out to me for a full write-up!
