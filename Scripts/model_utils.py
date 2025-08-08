import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import os


# ===== Dataset
class CO2Dataset(Dataset):
    def __init__(self, df, feature_cols, target_cols):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_cols].values.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===== Data Loading
def load_data(path, feature_cols, target_cols, test_size=0.3, val_size=0.4):
    df = pd.read_csv(path)

    print(f"Loaded dataset with {len(df)} samples and {len(df.columns)} columns.")
    # Drop lines with NaN values
    df = df.dropna(subset=feature_cols + target_cols)
    print(f"Dataset after dropping NaNs: {len(df)} samples.")

    # Ensure all feature columns are present
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' not found in the dataset.")
    # Ensure all target columns are present
    for col in target_cols:
        if col not in df.columns:
            raise ValueError(f"Target column '{col}' not found in the dataset.")
    # Ensure no NaN values in features or targets
    if df[feature_cols].isnull().any().any():
        print("Warning: NaN value found in line {} of feature columns.".format(
            df[feature_cols].isnull().any(axis=1).idxmax()))
        raise ValueError("NaN values found in feature columns.")
    if df[target_cols].isnull().any().any():
        print("Warning: NaN value found in line {} of target columns.".format(
            df[target_cols].isnull().any(axis=1).idxmax()))
        raise ValueError("NaN values found in target columns.")

    label_encoders = {}
    for col in target_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"Target column '{col}': {len(le.classes_)} unique classes")

    # Split the dataset into train, validation, and test sets    
    train_df, temp_df = train_test_split(df, test_size=test_size + val_size, 
                                        stratify=df[target_cols[0]], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_size / (test_size + val_size), 
                                      stratify=temp_df[target_cols[0]], random_state=42)

    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    # Scale the feature columns
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Convert label_encoders to mapper format for compatibility with existing code
    target_mappers = {}
    for col in target_cols:
        le = label_encoders[col]
        # Create mapping from original value to encoded value
        target_mappers[col] = {original: encoded for encoded, original in enumerate(le.classes_)}

    return train_df, val_df, test_df, scaler, target_mappers


# ===== Model
class CO2QuantumClassifier(nn.Module):
    def __init__(self, input_dim, output_dims):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.heads = nn.ModuleList([nn.Linear(64, out_dim) for out_dim in output_dims])

    def forward(self, x):
        x = self.shared(x)
        return [head(x) for head in self.heads]


# ===== Loss with l2 constraint penalty
def compute_loss(outputs, targets, criterion_list, penalty_weight=1.0):
    total_loss = 0
    batch_size = targets.shape[0]

    # Main CE loss
    for i in range(len(outputs)):
        total_loss += criterion_list[i](outputs[i], targets[:, i])

    # Apply constraint on l2 (index 2) based on v2 (index 1)
    v2_pred = torch.argmax(outputs[1], dim=1)    # predicted v2
    l2_pred = torch.argmax(outputs[2], dim=1)    # predicted l2

    # Valid l2s for each v2
    penalty = 0
    for i in range(batch_size):
        v2 = v2_pred[i].item()
        l2 = l2_pred[i].item()

        valid_l2s = list(range(v2, -1, -2)) if v2 % 2 == 0 else list(range(v2, 0, -2))
        if l2 not in valid_l2s:
            penalty += 1.0  # flat penalty per violation

    total_loss += penalty_weight * (penalty / batch_size)
    return total_loss


# ===== Train/Eval
def train(model, dataloader, optimizer, criterion_list, device):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = compute_loss(outputs, y, criterion_list)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion_list, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = compute_loss(outputs, y, criterion_list)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# ===== Outputs
def get_predictions(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X)
            preds = [torch.argmax(out, dim=1).cpu().numpy() for out in outputs]
            y_pred.append(np.stack(preds, axis=1))
            y_true.append(y.numpy())
    return np.vstack(y_true), np.vstack(y_pred)


def get_feature_importance(model, dataloader, criterion_list, device, feature_cols, target_cols, output_dir=None):
    model.eval()
    
    # Calculate baseline accuracy for each target
    y_true, y_pred = get_predictions(model, dataloader, device)
    baseline_accuracies = {label: accuracy_score(y_true[:, i], y_pred[:, i]) for i, label in enumerate(target_cols)}

    importance_results = {}
    for target_idx, target_label in enumerate(target_cols):
        print(f"\nCalculating feature importance for {target_label}...")
        feature_importance = {}
        with torch.no_grad():
            for feature_idx, feature_name in enumerate(feature_cols):
                # Create a copy of the original dataloader's data
                X_orig, y_orig = next(iter(dataloader))
                X_orig, y_orig = X_orig.to(device), y_orig.to(device)

                # Permute the values of one feature column
                X_permuted = X_orig.clone()
                idx = torch.randperm(X_permuted.size(0))
                X_permuted[:, feature_idx] = X_permuted[idx, feature_idx]

                # Get predictions with permuted data
                outputs = model(X_permuted)
                
                # Calculate accuracy for the permuted data for the current target
                permuted_preds = torch.argmax(outputs[target_idx], dim=1).cpu().numpy()
                permuted_accuracy = accuracy_score(y_orig[:, target_idx].cpu().numpy(), permuted_preds)
                
                # Calculate the drop in accuracy
                importance = baseline_accuracies[target_label] - permuted_accuracy
                feature_importance[feature_name] = importance
        
        importance_results[target_label] = feature_importance

    df = pd.DataFrame(importance_results)

    if output_dir:
        # Ensure the Plots directory exists
        os.makedirs(os.path.join(output_dir, "Plots"), exist_ok=True)
        
        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12), sharex=False)
        axes = axes.flatten()

        # Plot the feature importance for each quantum number
        for i, col in enumerate(df.columns):
            # Sort the features by importance for the current quantum number
            sorted_df = df[[col]].sort_values(by=col, ascending=True)
            
            # Plotting on the current axis
            ax = axes[i]
            sorted_df.plot(kind='barh', ax=ax, legend=False)
            ax.set_title(f"Feature Importance for {col}")
            ax.set_xlabel("Importance Score (Drop in Accuracy)")
            ax.set_ylabel("Feature")
            
        # Adjust layout to prevent overlapping titles and labels
        plt.tight_layout()

        # Save the combined plot to the output directory
        plt.savefig(os.path.join(output_dir, "Plots/feature_importance_plots.png"))

    return df


def save_accuracy_report(y_true, y_pred, target_cols, output_dir):
    # Ensure the CSVs directory exists
    os.makedirs(os.path.join(output_dir, "CSVs"), exist_ok=True)
    
    metrics = {}
    for i, label in enumerate(target_cols):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        metrics[label] = acc
        print(f"{label} Accuracy: {acc:.4f}")

    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(os.path.join(output_dir, "CSVs/test_accuracy.csv"), index=False)


def plot_confusion_matrices(y_true, y_pred, target_cols, output_dir):
    # Ensure the Plots directory exists
    os.makedirs(os.path.join(output_dir, "Plots"), exist_ok=True)
    
    for i, label in enumerate(target_cols):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix: {label}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Plots/confusion_{label}.png"))
        plt.close()


def plot_loss(train_losses, val_losses, output_dir):
    # Ensure the Plots directory exists
    os.makedirs(os.path.join(output_dir, "Plots"), exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Plots/loss_plot.png"))
    plt.close()