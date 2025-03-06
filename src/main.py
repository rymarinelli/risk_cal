import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer


from data_loader import load_prompt_injection_data, load_harmfulqa_data
from news import main as news_main
from preprocessing import compute_societal_risk 

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

class SocietalRiskDataset(Dataset):
    def __init__(self, dataframe, tokenizer, sorted_categories, use_categorical, max_length=128):
        """
        Args:
            dataframe (pd.DataFrame): Must contain "text", "label", and "societal risk".
                If use_categorical is True, it must also contain columns "distance_<Category>".
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for converting text to token IDs.
            sorted_categories (list): List of harmful categories (for selecting distance columns).
            use_categorical (bool): Whether per-category distance features are available.
            max_length (int): Maximum sequence length.
        """
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_categorical = use_categorical
        if self.use_categorical:
            self.distance_columns = [f"distance_{cat}" for cat in sorted_categories if f"distance_{cat}" in self.data.columns]
        else:
            self.distance_columns = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]
        label = row["label"]
        societal_risk = row["societal risk"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["label"] = torch.tensor(label, dtype=torch.long)
        encoding["societal_risk"] = torch.tensor(societal_risk, dtype=torch.float)
        if self.use_categorical:
            distance_values = [row[col] for col in self.distance_columns]
            encoding["distance_features"] = torch.tensor(distance_values, dtype=torch.float)
        encoding["text"] = row["text"]
        return encoding


def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_model_by_category(model, dataloader, device, sorted_categories):
    """
    Evaluate the model by assigning each sample to a category based on the smallest
    distance feature and computing per-category accuracy.
    """
    model.eval()
    all_true, all_pred, assigned_categories = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            distance_features = batch["distance_features"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=1)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predictions.cpu().numpy())
            # assign the category corresponding to the smallest distance.
            for dist in distance_features:
                min_index = torch.argmin(dist).item()
                assigned_categories.append(sorted_categories[min_index])
    df_eval = pd.DataFrame({
        "true": np.array(all_true),
        "pred": np.array(all_pred),
        "assigned_category": assigned_categories
    })
    df_eval["correct"] = (df_eval["true"] == df_eval["pred"]).astype(int)
    category_accuracy = df_eval.groupby("assigned_category")["correct"].mean()
    return category_accuracy, df_eval


def main(args):
    logger.info("Fetching and processing news data...")
    news_df = news_main()  
    if news_df.empty:
        logger.error("No news data available. Exiting.")
        return
    news_distribution = news_df["Matched Category"].value_counts(normalize=True).to_dict()
    logger.info("Harmful news distribution from news_df:")
    logger.info(news_distribution)


    logger.info("Loading prompt injection and HarmfulQA datasets...")
    prompt_df_train, prompt_df_test = load_prompt_injection_data()
    harmful_df = load_harmfulqa_data()
    categories = harmful_df["Category"].unique()
    sorted_categories = sorted(list(categories))
    logger.info(f"Identified categories: {sorted_categories}")

    #Compute societal risk 
    logger.info("Computing societal risk" + (" and per-category distances..." if args.use_categorical else "..."))
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    prompt_df_train = compute_societal_risk(prompt_df_train, harmful_df, embed_model, use_categorical=args.use_categorical)
    prompt_df_test = compute_societal_risk(prompt_df_test, harmful_df, embed_model, use_categorical=args.use_categorical)


    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = SocietalRiskDataset(prompt_df_train, tokenizer, sorted_categories, args.use_categorical, max_length=128)
    test_dataset = SocietalRiskDataset(prompt_df_test, tokenizer, sorted_categories, args.use_categorical, max_length=128)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    logger.info("DataLoaders created.")

    # Initialize model, hyperparameters, compute p_vector.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cal_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    cal_model.to(device)
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    alpha = 1.0       # Exponential scaling factor.
    lambda1 = 0.5     # Weight for composite calibration error loss.
    lambda2 = 0.5     # Weight for societal risk loss.
    L_Risk_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cal_model.parameters(), lr=learning_rate)

    if args.use_categorical:
        p_values = [news_distribution.get(cat, 0.0) for cat in sorted_categories]
        p_vector = torch.tensor(p_values, dtype=torch.float, device=device)
        logger.info("Sorted harmful news distribution:")
        logger.info({cat: news_distribution.get(cat, 0.0) for cat in sorted_categories})
    else:
        p_vector = None

    logger.info("Starting training...")
    for epoch in range(num_epochs):
        cal_model.train()
        epoch_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            societal_scores = batch["societal_risk"].to(device)
            outputs = cal_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs["logits"]
            L_Risk = L_Risk_fn(logits, labels)
            if args.loss_mode == "standard":
                loss = L_Risk
            elif args.loss_mode == "calibrated":
                probs = torch.softmax(logits, dim=1)
                pred_prob = probs[:, 1] 
                sample_calibration_error = (pred_prob - labels.float()) ** 2
                L_CaliError = torch.mean(societal_scores * sample_calibration_error)
                if args.use_categorical:
                    distance_features = batch["distance_features"].to(device)
                    revised_risk = torch.sum(p_vector * torch.exp(-alpha * distance_features), dim=1)
                    L_SocietalRisk = torch.mean(revised_risk)
                else:
                    uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    f_uncertainty = torch.exp(alpha * uncertainty)
                    L_SocietalRisk = torch.mean(societal_scores * f_uncertainty)
                loss = L_Risk + lambda1 * L_CaliError + lambda2 * L_SocietalRisk
            else:
                raise ValueError(f"Unsupported loss_mode: {args.loss_mode}")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")

    final_accuracy = evaluate_model(cal_model, test_dataloader, device)
    logger.info("Final Test Accuracy:")
    logger.info(f"Calibrated Model Accuracy: {final_accuracy:.4f}")

    if not args.use_categorical:
        # Global Centroid Evaluation with Risk Quartiles
        cal_model.eval()
        all_true, all_pred, all_risk, all_texts = [], [], [], []
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                societal_scores = batch["societal_risk"].to(device)
                texts = batch.get("text", None)
                if texts is not None:
                    all_texts.extend(texts)
                outputs = cal_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                logits = outputs["logits"]
                predictions = torch.argmax(logits, dim=1)
                all_true.append(labels.cpu().numpy())
                all_pred.append(predictions.cpu().numpy())
                all_risk.append(societal_scores.cpu().numpy())
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        all_risk = np.concatenate(all_risk)
        overall_accuracy = np.mean(all_true == all_pred)
        weighted_accuracy = np.sum(all_risk * (all_true == all_pred)) / np.sum(all_risk)
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"Weighted Accuracy: {weighted_accuracy:.4f}")
        df_eval = pd.DataFrame({
            "true": all_true,
            "pred": all_pred,
            "risk": all_risk
        })
        if len(all_texts) == len(df_eval):
            df_eval["text"] = all_texts
        df_eval["quartile"] = pd.qcut(df_eval["risk"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        quartile_accuracies = {}
        for quartile, group in df_eval.groupby("quartile", observed=True):
            quartile_accuracies[quartile] = np.mean(group["true"] == group["pred"])
        print("\nAccuracy by Risk Quartiles:")
        for quartile, acc in quartile_accuracies.items():
            count = len(df_eval[df_eval["quartile"] == quartile])
            print(f"Quartile {quartile}: {count} samples, Accuracy: {acc:.4f}")
        x = np.arange(4)
        quartiles = ["Q1", "Q2", "Q3", "Q4"]
        acc_values = [quartile_accuracies.get(q, np.nan) for q in quartiles]
        plt.figure(figsize=(8, 6))
        plt.bar(x, acc_values, width=0.6, label='Model')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Risk Quartile (Global Centroid)')
        plt.xticks(x, quartiles)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        # Categorical Evaluation: Per-Category Accuracy.
        baseline_category_accuracy, df_eval_base = evaluate_model_by_category(
            cal_model, test_dataloader, device, sorted_categories
        )
        calibrated_category_accuracy, df_eval_cal = evaluate_model_by_category(
            cal_model, test_dataloader, device, sorted_categories
        )
        # Get the list of categories.
        categories_list = baseline_category_accuracy.index.tolist()
        baseline_acc = [baseline_category_accuracy.get(cat, np.nan) for cat in categories_list]
        calibrated_acc = [calibrated_category_accuracy.get(cat, np.nan) for cat in categories_list]
        x = np.arange(len(categories_list))
        width = 0.35
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, baseline_acc, width, label='Baseline Model', color='skyblue')
        plt.bar(x + width/2, calibrated_acc, width, label='Calibrated Model', color='lightgreen')
        plt.xlabel('Harmful Category')
        plt.ylabel('Accuracy')
        plt.title('Per-Category Accuracy Comparison')
        plt.xticks(x, categories_list, rotation=45)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.show()
        # Compute difference.
        baseline_category_accuracy = baseline_category_accuracy.reindex(sorted_categories)
        calibrated_category_accuracy = calibrated_category_accuracy.reindex(sorted_categories)
        accuracy_difference = calibrated_category_accuracy - baseline_category_accuracy
        categories_list = accuracy_difference.index.tolist()
        x = np.arange(len(categories_list))
        colors = ['green' if diff >= 0 else 'red' for diff in accuracy_difference]
        plt.figure(figsize=(10, 6))
        plt.bar(x, accuracy_difference.values, color=colors, width=0.6)
        plt.xlabel("Harmful Category")
        plt.ylabel("Accuracy Difference (Calibrated - Baseline)")
        plt.title("Per-Category Accuracy Difference")
        plt.xticks(x, categories_list, rotation=45)
        plt.axhline(0, color='black', linewidth=1)
        plt.tight_layout()
        plt.show()

    os.makedirs("saved_models", exist_ok=True)
    cal_model.save_pretrained(os.path.join("saved_models", "calibrated_model"))
    tokenizer.save_pretrained(os.path.join("saved_models", "model_tokenizer"))
    logger.info("Models and tokenizer saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk_Cal Project: Train and Evaluate Risk-Based Models")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train", help="Pipeline mode")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--model_name", type=str, default="protectai/deberta-v3-base-prompt-injection-v2", help="Model name or path")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for DataLoaders")
    parser.add_argument("--loss_mode", type=str, choices=["standard", "calibrated"], default="calibrated",
                        help="Choose loss function mode: 'standard' uses cross-entropy only; 'calibrated' adds calibration and societal risk losses.")
    parser.add_argument("--use_categorical", action="store_true", help="If set, compute and use per-category distance features; otherwise, use only global harm.")
    args = parser.parse_args()
    
    main(args)
