import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from news import main as news_main


def setup_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_harmful_data(embedder):
    harmful_df = pd.read_json("hf://datasets/declare-lab/CategoricalHarmfulQA/data/catqa_english.json", lines=True)
    questions = harmful_df['Question'].tolist()
    embeddings = embedder.encode(questions)
    harmful_df['embedding'] = list(embeddings)
    global_centroid = np.mean(np.vstack(harmful_df['embedding'].values), axis=0)
    categories = harmful_df['Category'].unique()
    category_centroids = {cat: np.mean(np.vstack(harmful_df[harmful_df['Category'] == cat]['embedding'].values), axis=0)
                          for cat in categories}
    harmful_df.drop(columns=['embedding'], inplace=True)
    return harmful_df, global_centroid, category_centroids


def load_prompt_injection_data(embedder, global_centroid, category_centroids):
    train_df = pd.read_parquet("hf://datasets/deepset/prompt-injections/data/train-00000-of-00001-9564e8b05b4757ab.parquet")
    test_df = pd.read_parquet("hf://datasets/deepset/prompt-injections/data/test-00000-of-00001-701d16158af87368.parquet")
    
    for df in [train_df, test_df]:
        texts = df["text"].tolist()
        embeddings = embedder.encode(texts)
        df["embedding"] = list(embeddings)
        df["societal risk"] = df["embedding"].apply(lambda emb: np.linalg.norm(emb - global_centroid))
        for cat, centroid in category_centroids.items():
            df[f"distance_{cat}"] = df["embedding"].apply(lambda emb: np.linalg.norm(emb - centroid))
        df.drop(columns=["embedding"], inplace=True)
    return train_df, test_df


class InjectionDatasetCategorical(Dataset):
    def __init__(self, dataframe, tokenizer, sorted_categories, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.distance_columns = [f"distance_{cat}" for cat in sorted_categories if f"distance_{cat}" in self.data.columns]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(row["text"], truncation=True, padding="max_length",
                                  max_length=self.max_length, return_tensors="pt")
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["label"] = torch.tensor(row["label"], dtype=torch.long)
        encoding["societal_risk"] = torch.tensor(row["societal risk"], dtype=torch.float)
        encoding["distance_features"] = torch.tensor([row[col] for col in self.distance_columns], dtype=torch.float)
        return encoding


class InjectionDatasetGlobal(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(row["text"], truncation=True, padding="max_length",
                                  max_length=self.max_length, return_tensors="pt")
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["label"] = torch.tensor(row["label"], dtype=torch.long)
        encoding["societal_risk"] = torch.tensor(row["societal risk"], dtype=torch.float)
        encoding["text"] = row["text"]
        return encoding


def get_dataloaders(tokenizer, train_df, test_df, use_categorical, batch_size=16, max_length=128):
    if use_categorical:
        cats = sorted(list(set(train_df["Matched Category"].unique())))
        train_dataset = InjectionDatasetCategorical(train_df, tokenizer, cats, max_length)
        test_dataset = InjectionDatasetCategorical(test_df, tokenizer, cats, max_length)
    else:
        train_dataset = InjectionDatasetGlobal(train_df, tokenizer, max_length)
        test_dataset = InjectionDatasetGlobal(test_df, tokenizer, max_length)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def evaluate_model_by_category(model, dataloader, device, sorted_categories):
    model.eval()
    all_true, all_pred, assigned = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            distances = batch["distance_features"].to(device)
            logits = model(input_ids=input_ids, attention_mask=mask, return_dict=True)["logits"]
            preds = torch.argmax(logits, dim=1)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
            for dist in distances:
                assigned.append(sorted_categories[torch.argmin(dist).item()])
    df = pd.DataFrame({"true": np.array(all_true), "pred": np.array(all_pred), "assigned_category": assigned})
    df["correct"] = (df["true"] == df["pred"]).astype(int)
    return df.groupby("assigned_category")["correct"].mean(), df


def evaluate_model_global(model, dataloader, device):
    model.eval()
    all_true, all_pred, all_risk = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            risk = batch["societal_risk"].to(device)
            logits = model(input_ids=input_ids, attention_mask=mask, return_dict=True)["logits"]
            preds = torch.argmax(logits, dim=1)
            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_risk.append(risk.cpu().numpy())
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    all_risk = np.concatenate(all_risk)
    overall = np.mean(all_true == all_pred)
    weighted = np.sum(all_risk * (all_true == all_pred)) / np.sum(all_risk)
    df = pd.DataFrame({"true": all_true, "pred": all_pred, "risk": all_risk})
    try:
        df["quartile"] = pd.qcut(df["risk"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    except Exception:
        df["quartile"] = "Constant"
    quartiles = {}
    if df["quartile"].dtype == object:
        quartiles["Constant"] = overall
    else:
        for q, group in df.groupby("quartile", observed=True):
            quartiles[q] = np.mean(group["true"] == group["pred"])
    return overall, weighted, quartiles, df


def print_comparison_table(base_overall, base_weighted, base_quartiles,
                           cal_overall, cal_weighted, cal_quartiles):
    data = {
        "Metric": ["Overall Accuracy", "Weighted Accuracy", "Accuracy (Q1 – Lowest Risk)",
                   "Accuracy (Q2)", "Accuracy (Q3)", "Accuracy (Q4 – Highest Risk)"],
        "Base Model": [f"{base_overall*100:.2f}%", f"{base_weighted*100:.2f}%",
                       f"{base_quartiles.get('Q1', np.nan)*100:.2f}%", f"{base_quartiles.get('Q2', np.nan)*100:.2f}%",
                       f"{base_quartiles.get('Q3', np.nan)*100:.2f}%", f"{base_quartiles.get('Q4', np.nan)*100:.2f}%"],
        "Calibrated Model": [f"{cal_overall*100:.2f}%", f"{cal_weighted*100:.2f}%",
                             f"{cal_quartiles.get('Q1', np.nan)*100:.2f}%", f"{cal_quartiles.get('Q2', np.nan)*100:.2f}%",
                             f"{cal_quartiles.get('Q3', np.nan)*100:.2f}%", f"{cal_quartiles.get('Q4', np.nan)*100:.2f}%"]
    }
    df = pd.DataFrame(data)
    print("\nComparison of Base vs. Calibrated Models (Global Metrics):")
    print(df)


def plot_category_comparison(base_acc, cal_acc, sorted_categories, filename="category_comparison.png"):
    base = base_acc.reindex(sorted_categories)
    cal = cal_acc.reindex(sorted_categories)
    cats = base.index.tolist()
    x = np.arange(len(cats))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, [base.get(c, np.nan) for c in cats], width, label="Baseline", color="skyblue")
    plt.bar(x + width/2, [cal.get(c, np.nan) for c in cats], width, label="Calibrated", color="lightgreen")
    plt.xlabel("Harmful Category")
    plt.ylabel("Accuracy")
    plt.title("Per-Category Accuracy Comparison")
    plt.xticks(x, cats, rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_accuracy_difference(base_acc, cal_acc, sorted_categories, filename="accuracy_difference.png"):
    diff = cal_acc.reindex(sorted_categories) - base_acc.reindex(sorted_categories)
    cats = diff.index.tolist()
    x = np.arange(len(cats))
    colors = ["green" if d >= 0 else "red" for d in diff.values]
    plt.figure(figsize=(10, 6))
    plt.bar(x, diff.values, color=colors, width=0.6)
    plt.xlabel("Harmful Category")
    plt.ylabel("Difference (Calibrated - Baseline)")
    plt.title("Per-Category Accuracy Difference")
    plt.xticks(x, cats, rotation=45)
    plt.axhline(0, color="black", linewidth=1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_accuracy_by_quartile(base_quartiles, cal_quartiles, filename="accuracy_by_quartile.png"):
    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    base_vals = [base_quartiles.get(q, np.nan) for q in quartiles]
    cal_vals = [cal_quartiles.get(q, np.nan) for q in quartiles]
    x = np.arange(len(quartiles))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, base_vals, width, label="Baseline")
    ax.bar(x + width/2, cal_vals, width, label="Calibrated")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Risk Quartile")
    ax.set_xticks(x)
    ax.set_xticklabels(quartiles)
    ax.legend()
    plt.savefig(filename)
    plt.show()


def plot_overall_accuracy(base_overall, base_weighted, cal_overall, cal_weighted, filename="overall_accuracy.png"):
    labels = ["Overall Accuracy", "Weighted Accuracy"]
    base_vals = [base_overall, base_weighted]
    cal_vals = [cal_overall, cal_weighted]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, base_vals, width, label="Baseline")
    ax.bar(x + width/2, cal_vals, width, label="Calibrated")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig(filename)
    plt.show()


def train_base_model(model, dataloader, device, num_epochs, lr, early_stop_threshold):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            loss = criterion(model(input_ids=input_ids, attention_mask=mask, return_dict=True)["logits"], labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Base Epoch {epoch+1}: Loss {avg_loss:.4f}")
        if avg_loss < early_stop_threshold:
            break
    return model


def train_calibrated_model_categorical(model, dataloader, device, num_epochs, lr, alpha, lambda1, lambda2, p_vector, early_stop_threshold):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            distances = batch["distance_features"].to(device)
            logits = model(input_ids=input_ids, attention_mask=mask, return_dict=True)["logits"]
            L_Risk = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            L_CaliError = torch.mean((probs[:, 1] - labels.float()) ** 2)
            L_SocietalRisk = torch.mean(torch.sum(p_vector * torch.exp(-alpha * distances), dim=1))
            loss = L_Risk + lambda1 * L_CaliError + lambda2 * L_SocietalRisk
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Calibrated (Cat) Epoch {epoch+1}: Loss {avg_loss:.4f}")
        if avg_loss < early_stop_threshold:
            break
    return model


def train_calibrated_model_non_categorical(model, dataloader, device, num_epochs, lr, alpha, lambda1, lambda2, early_stop_threshold):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            risk = batch["societal_risk"].to(device)
            logits = model(input_ids=input_ids, attention_mask=mask, return_dict=True)["logits"]
            L_Risk = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            L_CaliError = torch.mean(risk * ((probs[:, 1] - labels.float()) ** 2))
            uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            L_SocietalRisk = torch.mean(risk * torch.exp(alpha * uncertainty))
            loss = L_Risk + lambda1 * L_CaliError + lambda2 * L_SocietalRisk
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Calibrated (Non-Cat) Epoch {epoch+1}: Loss {avg_loss:.4f}")
        if avg_loss < early_stop_threshold:
            break
    return model


def main_categorical(args):
    device = setup_device()
    model_name = "protectai/deberta-v3-base-prompt-injection-v2"
    num_epochs, lr = args.num_epochs, 2e-5
    alpha, lambda1, lambda2 = 1.0, 0.5, 0.5

    news_df = news_main()
    news_dist = news_df["Matched Category"].value_counts(normalize=True).to_dict()

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    harmful_df, global_centroid, category_centroids = load_harmful_data(embedder)
    cats = sorted(list(np.unique(harmful_df["Category"])))
    train_df, test_df = load_prompt_injection_data(embedder, global_centroid, category_centroids)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader = DataLoader(InjectionDatasetCategorical(train_df, tokenizer, cats, max_length=128),
                              batch_size=2, shuffle=True)
    test_loader  = DataLoader(InjectionDatasetCategorical(test_df, tokenizer, cats, max_length=128),
                              batch_size=2, shuffle=False)

    p_vector = torch.tensor([news_dist.get(cat, 0.0) for cat in cats], dtype=torch.float, device=device)
    print("Ordered News Distribution:", {cat: news_dist.get(cat, 0.0) for cat in cats})

    baseline_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    cal_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    print("Training Baseline (Categorical)...")
    baseline_model = train_base_model(baseline_model, train_loader, device, num_epochs, lr, 0.05)
    print("Training Calibrated (Categorical)...")
    cal_model = train_calibrated_model_categorical(cal_model, train_loader, device, num_epochs, lr, alpha, lambda1, lambda2, p_vector, 0.05)

    base_cat_acc, _ = evaluate_model_by_category(baseline_model, test_loader, device, cats)
    cal_cat_acc, _ = evaluate_model_by_category(cal_model, test_loader, device, cats)
    print("Per-Category Accuracy (Baseline):", base_cat_acc)
    print("Per-Category Accuracy (Calibrated):", cal_cat_acc)
    plot_category_comparison(base_cat_acc, cal_cat_acc, cats, filename="category_comparison.png")
    plot_accuracy_difference(base_cat_acc, cal_cat_acc, cats, filename="accuracy_difference.png")

    base_overall, base_weighted, base_quartiles, _ = evaluate_model_global(baseline_model, test_loader, device)
    cal_overall, cal_weighted, cal_quartiles, _ = evaluate_model_global(cal_model, test_loader, device)
    print_comparison_table(base_overall, base_weighted, base_quartiles, cal_overall, cal_weighted, cal_quartiles)
    plot_accuracy_by_quartile(base_quartiles, cal_quartiles, filename="accuracy_by_quartile.png")
    plot_overall_accuracy(base_overall, base_weighted, cal_overall, cal_weighted, filename="overall_accuracy.png")

    baseline_model.save_pretrained("baseline_model")
    cal_model.save_pretrained("calibrated_model")
    tokenizer.save_pretrained("model_tokenizer")


def main_non_categorical(args):
    device = setup_device()
    model_name = "protectai/deberta-v3-base-prompt-injection-v2"
    num_epochs, lr = args.num_epochs, 2e-5
    alpha, lambda1, lambda2 = 1.0, 0.5, 0.5

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    harmful_df, global_centroid, _ = load_harmful_data(embedder)
    train_df, test_df = load_prompt_injection_data(embedder, global_centroid, {})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader, test_loader = get_dataloaders(tokenizer, train_df, test_df, use_categorical=False, batch_size=2, max_length=128)

    baseline_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    cal_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    print("Training Baseline (Non-Categorical)...")
    baseline_model = train_base_model(baseline_model, train_loader, device, num_epochs, lr, 0.05)
    print("Training Calibrated (Non-Categorical)...")
    cal_model = train_calibrated_model_non_categorical(cal_model, train_loader, device, num_epochs, lr, alpha, lambda1, lambda2, 0.05)

    base_overall, base_weighted, base_quartiles, _ = evaluate_model_global(baseline_model, test_loader, device)
    cal_overall, cal_weighted, cal_quartiles, _ = evaluate_model_global(cal_model, test_loader, device)
    print_comparison_table(base_overall, base_weighted, base_quartiles, cal_overall, cal_weighted, cal_quartiles)
    plot_accuracy_by_quartile(base_quartiles, cal_quartiles, filename="accuracy_by_quartile_noncat.png")
    plot_overall_accuracy(base_overall, base_weighted, cal_overall, cal_weighted, filename="overall_accuracy_noncat.png")

    baseline_model.save_pretrained("baseline_model_noncat")
    cal_model.save_pretrained("calibrated_model_noncat")
    tokenizer.save_pretrained("model_tokenizer_noncat")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and Evaluate Risk-Based Models")
    parser.add_argument("--use_categorical", action="store_true",
                        help="Use per-category distance features. If not set, non-categorical mode is used.")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs (default: 1)")
    args = parser.parse_args()
    if args.use_categorical:
        main_categorical(args)
    else:
        main_non_categorical(args)
