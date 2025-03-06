# src/data_loader.py
import pandas as pd

def load_prompt_injection_data():
    splits = {
        'train': 'data/train-00000-of-00001-9564e8b05b4757ab.parquet',
        'test': 'data/test-00000-of-00001-701d16158af87368.parquet'
    }
    train_df = pd.read_parquet("hf://datasets/deepset/prompt-injections/" + splits["train"])
    test_df = pd.read_parquet("hf://datasets/deepset/prompt-injections/" + splits["test"])
    return train_df, test_df

def load_harmfulqa_data():
    splits = {'en': 'data/catqa_english.json'}
    harmful_df = pd.read_json("hf://datasets/declare-lab/CategoricalHarmfulQA/" + splits["en"], lines=True)
    return harmful_df
