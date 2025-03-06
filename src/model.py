import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn
import torch.optim as optim

def load_tokenizer_and_model(model_name):
    """
    Loads the tokenizer and model from the Hugging Face hub.

    Args:
        model_name (str): The model identifier (e.g., "protectai/deberta-v3-base-prompt-injection-v2").

    Returns:
        tokenizer, model: The loaded tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def train_model(model, dataloader, num_epochs, learning_rate, device):
    """
    Trains the model on data provided by the dataloader.

    Args:
        model: The PyTorch model to train.
        dataloader: A DataLoader providing training batches.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        device (str): 'cuda' or 'cpu'.

    Returns:
        model: The trained model.
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs["logits"]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    
    return model

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the provided data.

    Args:
        model: The trained model.
        dataloader: A DataLoader providing evaluation batches.
        device (str): 'cuda' or 'cpu'.

    Returns:
        accuracy (float): The evaluation accuracy.
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
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

def save_model(model, tokenizer, output_dir):
    """
    Saves the model and tokenizer to the specified directory.

    Args:
        model: The trained model.
        tokenizer: The associated tokenizer.
        output_dir (str): The directory to save the model and tokenizer.
    """
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
