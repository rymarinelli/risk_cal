import numpy as np

def compute_societal_risk(prompt_df, harmful_df, embed_model, use_categorical=False):
    """
    Computes the societal risk for each prompt injection sample based on its distance
    to the global centroid (computed from HarmfulQA question embeddings). Optionally,
    computes per-category distance columns if use_categorical is True.

    Args:
        prompt_df (pd.DataFrame): DataFrame of prompt injection samples. Must contain a "text" column.
        harmful_df (pd.DataFrame): DataFrame of HarmfulQA samples. Must contain "Question" and "Category" columns.
        embed_model (SentenceTransformer): Embedding model for generating sentence embeddings.
        use_categorical (bool, optional): If True, computes per-category distance columns. Defaults to False.

    Returns:
        prompt_df (pd.DataFrame): The original DataFrame with added "societal risk" and, if enabled,
                                  "distance_<Category>" columns.
    """
    
    harmful_questions = harmful_df['Question'].tolist()
    harmful_embeddings = embed_model.encode(harmful_questions)
    harmful_df['embedding'] = list(harmful_embeddings)
    
    # Compute the global centroid from harmful embeddings.
    global_centroid = np.mean(np.vstack(harmful_df['embedding'].values), axis=0)
    
    
    prompt_texts = prompt_df['text'].tolist()
    prompt_embeddings = embed_model.encode(prompt_texts)
    prompt_df['embedding'] = list(prompt_embeddings)
    
    # Compute societal risk as the Euclidean distance from each prompt to the global centroid.
    prompt_df['societal risk'] = prompt_df['embedding'].apply(lambda emb: np.linalg.norm(emb - global_centroid))
    
    if use_categorical:
        # Compute per-category distance columns.
        categories = harmful_df['Category'].unique()
        for cat in categories:
            # Compute the centroid for the current category.
            cat_embeddings = np.vstack(harmful_df[harmful_df['Category'] == cat]['embedding'].values)
            centroid = np.mean(cat_embeddings, axis=0)
            # Compute the Euclidean distance from each prompt to the category centroid.
            prompt_df[f"distance_{cat}"] = prompt_df['embedding'].apply(lambda emb: np.linalg.norm(emb - centroid))
    
    prompt_df.drop(columns=['embedding'], inplace=True)
    harmful_df.drop(columns=['embedding'], inplace=True)
    
    return prompt_df
