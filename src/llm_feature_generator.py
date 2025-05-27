import pandas as pd
import numpy as np
from transformers import pipeline

def generate_synthetic_news(df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic news snippets based on stock returns."""
    df['News'] = np.where(
        df['Returns'] > 0,
        ["Positive market outlook for stock due to strong earnings."] * len(df),
        ["Concerns arise over stock performance amid market volatility."] * len(df)
    )
    return df

def generate_llm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate sentiment scores using a pre-trained LLM (DistilBERT by default)."""
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        print(f"Error loading model: {e}. Falling back to random sentiment.")
        np.random.seed(42)
        df['Sentiment'] = np.random.uniform(0.0, 1.0, len(df))
        return df

    df = generate_synthetic_news(df)

    sentiments = []
    for news in df['News']:
        result = sentiment_analyzer(news)[0]
        score = result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']
        sentiments.append(score)

    df['Sentiment'] = sentiments
    return df
