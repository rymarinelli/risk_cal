import feedparser
import pandas as pd
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

def fetch_and_summarize_news(rss_feed_url):
    """
    Fetches news articles from an RSS feed and summarizes each article.

    Args:
        rss_feed_url (str): The URL of the RSS feed.

    Returns:
        news_df (pd.DataFrame): DataFrame containing columns for Title, Link, and Summary.
    """
    # Parse the RSS feed.
    feed = feedparser.parse(rss_feed_url)
    articles = feed.entries
    if not articles:
        print("No articles found from the RSS feed.")
        return pd.DataFrame()

    # summarization pipeline 
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    news_data = []
    for article in articles:
        title = article.get("title", "No Title")
        description = article.get("summary", "")
        link = article.get("link", "")
        # Combine title and description for better context.
        full_text = f"{title}. {description}"
        try:
            summary = summarizer(full_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        except Exception as e:
            summary = "Error summarizing this article."
        news_data.append({
            "Title": title,
            "Link": link,
            "Summary": summary
        })
    news_df = pd.DataFrame(news_data)
    return news_df

def load_harmfulqa_compute_centroids():
    """
    Loads the HarmfulQA dataset and computes category centroids based on the question embeddings.

    Returns:
        category_centroids (dict): Mapping from category to centroid embedding.
    """
    splits = {'en': 'data/catqa_english.json'}
    harmful_df = pd.read_json("hf://datasets/declare-lab/CategoricalHarmfulQA/" + splits["en"], lines=True)
    print("HarmfulQA dataset preview:")
    print(harmful_df[['Question', 'Category']].head())

  
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    harmful_questions = harmful_df['Question'].tolist()
    harmful_embeddings = embed_model.encode(harmful_questions)
    harmful_df['embedding'] = list(harmful_embeddings)

    categories = harmful_df['Category'].unique()
    category_centroids = {}
    for cat in categories:
        cat_embeddings = np.vstack(harmful_df[harmful_df['Category'] == cat]['embedding'].values)
        centroid = np.mean(cat_embeddings, axis=0)
        category_centroids[cat] = centroid

    harmful_df.drop(columns=['embedding'], inplace=True)
    return category_centroids

def compute_news_category_distances(news_df, category_centroids, embed_model):
    """
    Computes distances for each news summary to each category centroid and assigns a matched category.

    Args:
        news_df (pd.DataFrame): DataFrame containing news summaries.
        category_centroids (dict): Dictionary mapping category names to centroid embeddings.
        embed_model (SentenceTransformer): Model to generate embeddings for news summaries.

    Returns:
        news_df (pd.DataFrame): DataFrame with additional distance columns, the matched category, and matched distance.
    """
    # Compute embeddings for each news summary.
    news_texts = news_df["Summary"].tolist()
    news_embeddings = embed_model.encode(news_texts)
    news_df["Embedding"] = list(news_embeddings)

    # Compute the distance from each news summary to every category centroid.
    for cat, centroid in category_centroids.items():
        col_name = f"Distance_{cat}"
        news_df[col_name] = news_df["Embedding"].apply(lambda emb: np.linalg.norm(emb - centroid))

    # Determine the matched category based on the smallest distance.
    distance_cols = [f"Distance_{cat}" for cat in category_centroids.keys()]
    news_df["Matched Category"] = news_df[distance_cols].idxmin(axis=1).str.replace("Distance_", "")
    news_df["Matched Distance"] = news_df[distance_cols].min(axis=1)

    news_df.drop(columns=["Embedding"], inplace=True)
    return news_df

def main():
    """
    Main routine to fetch, summarize, and categorize news articles.
    """
    rss_feed_url = "http://feeds.bbci.co.uk/news/rss.xml"
    news_df = fetch_and_summarize_news(rss_feed_url)
    if news_df.empty:
        print("No news articles to process.")
        return
    print("News Summaries:")
    print(news_df.head(), "\n")

    category_centroids = load_harmfulqa_compute_centroids()

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')


    news_df = compute_news_category_distances(news_df, category_centroids, embed_model)
    print("News Summaries with Categorization Distances and Matched Category:")
    print(news_df.head())
    return news_df

if __name__ == "__main__":
    main()
