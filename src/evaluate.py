import torch
import numpy as np
import pandas as pd

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model overall.
    
    Args:
        model: The trained model.
        dataloader: DataLoader for evaluation.
        device (str): 'cuda' or 'cpu'.
    
    Returns:
        accuracy (float): Overall accuracy.
    """
    model.eval()
    total, correct = 0, 0
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
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def evaluate_model_by_category(model, dataloader, device, sorted_categories):
    """
    Evaluates the model by assigning each sample to a category based on the smallest
    distance feature and computing per-category accuracy.
    
    Args:
        model: The classification model.
        dataloader: DataLoader for evaluation. Must provide a "distance_features" field.
        device: 'cuda' or 'cpu'.
        sorted_categories (list): List of categories in the same order as the distance features.
        
    Returns:
        category_accuracy: Pandas Series with per-category accuracy.
        df_eval: DataFrame with true labels, predictions, and assigned categories for each sample.
    """
    model.eval()
    all_true, all_pred, assigned_categories = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            # "distance_features" (batch_size, num_categories)
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

def plot_category_accuracy(baseline_category_accuracy, calibrated_category_accuracy, sorted_categories):
    """
    Plots a comparison of per-category accuracy between baseline and calibrated models.
    
    Args:
        baseline_category_accuracy: Pandas Series with baseline model accuracies.
        calibrated_category_accuracy: Pandas Series with calibrated model accuracies.
        sorted_categories (list): List of categories.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure the same order of categories.
    baseline_category_accuracy = baseline_category_accuracy.reindex(sorted_categories)
    calibrated_category_accuracy = calibrated_category_accuracy.reindex(sorted_categories)
    
    categories = sorted_categories
    baseline_acc = [baseline_category_accuracy.get(cat, np.nan) for cat in categories]
    calibrated_acc = [calibrated_category_accuracy.get(cat, np.nan) for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, baseline_acc, width, label='Baseline Model')
    plt.bar(x + width/2, calibrated_acc, width, label='Calibrated Model')
    plt.xlabel('Harmful Category')
    plt.ylabel('Accuracy')
    plt.title('Per-Category Accuracy Comparison')
    plt.xticks(x, categories, rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_accuracy_difference(baseline_category_accuracy, calibrated_category_accuracy, sorted_categories):
    """
    Plots the difference in per-category accuracy (calibrated - baseline).
    
    Args:
        baseline_category_accuracy: Pandas Series with baseline model accuracies.
        calibrated_category_accuracy: Pandas Series with calibrated model accuracies.
        sorted_categories (list): List of categories.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure the same order.
    baseline_category_accuracy = baseline_category_accuracy.reindex(sorted_categories)
    calibrated_category_accuracy = calibrated_category_accuracy.reindex(sorted_categories)
    
    accuracy_difference = calibrated_category_accuracy - baseline_category_accuracy
    categories = sorted_categories
    x = np.arange(len(categories))
    
    # Use green for improvements and red for drops.
    colors = ['green' if diff >= 0 else 'red' for diff in accuracy_difference]
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, accuracy_difference.values, color=colors, width=0.6)
    plt.xlabel("Harmful Category")
    plt.ylabel("Accuracy Difference (Calibrated - Baseline)")
    plt.title("Per-Category Accuracy Difference")
    plt.xticks(x, categories, rotation=45)
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()
    plt.show()
