import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from plotting import plot_iso_by_energy_split, plot_energy_distributions_detailed, plot_feature_importance


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
def load_data(path,
        feature_cols,
        target_cols,
        energy_splitting=False,
        energy_col='E',
        train_size=0.3,
        val_size=0.3,
        test_size=0.4,
        overlap_fraction=0.1,
        random_state=42,
        output_dir=None,
        plot_energy_distributions=True
    ):
    """
    Load data with sequential energy-based splitting for distinct energy ranges.
    """
    df = pd.read_csv(path)
    print(f"Loaded dataset with {len(df)} samples and {len(df.columns)} columns.")
    
    # Drop lines with NaN values
    df = df.dropna(subset=feature_cols + target_cols)
    print(f"Dataset after dropping NaNs: {len(df)} samples.")
    
    # Validation checks
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' not found in the dataset.")
    for col in target_cols:
        if col not in df.columns:
            raise ValueError(f"Target column '{col}' not found in the dataset.")
    if df[feature_cols].isnull().any().any():
        raise ValueError("NaN values found in feature columns.")
    if df[target_cols].isnull().any().any():
        raise ValueError("NaN values found in target columns.")

    # Store original energy values before any processing
    original_energy_col = f'{energy_col}_original'
    df[original_energy_col] = df[energy_col].copy()

    # Target encoding
    label_encoders = {}
    for col in target_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"Target column '{col}': {len(le.classes_)} unique classes")

    if energy_splitting:
        # Sequential energy split using original energy values
        print(f"\nUsing sequential energy splitting...")
        train_df, val_df, test_df = sequential_energy_split(
            df, original_energy_col, 
            train_size=train_size, val_size=val_size, test_size=test_size,
            overlap_fraction=overlap_fraction, random_state=random_state
        )
    else:
        # Split the dataset into train, validation, and test sets    
        train_df, temp_df = train_test_split(df, test_size=test_size + val_size, 
                                            stratify=df[target_cols[0]], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=test_size / (test_size + val_size), 
                                        stratify=temp_df[target_cols[0]], random_state=42)

    print(f"\nFinal split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Plot splits of energy distributions by isotopologue
    plot_iso_by_energy_split(train_df, val_df, test_df, iso_col='iso', energy_col='E_original', n_col=3, output_dir=output_dir)
    train_df.drop(columns=['iso'], inplace=True)
    val_df.drop(columns=['iso'], inplace=True)
    test_df.drop(columns=['iso'], inplace=True)

    # Scale features (but keep original energy for plotting)
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Generate energy distribution plots if requested
    if plot_energy_distributions:
        print("\nGenerating energy distribution plots...")
        plot_energy_distributions_detailed(train_df, val_df, test_df, 
                                          energy_col=original_energy_col,
                                          output_dir=output_dir)

    # Create target mappers for compatibility
    target_mappers = {}
    for col in target_cols:
        le = label_encoders[col]
        target_mappers[col] = {original: encoded for encoded, original in enumerate(le.classes_)}

    return train_df, val_df, test_df, scaler, target_mappers


# ===== Sequential Energy Split
def sequential_energy_split(df,
                            energy_col='E',
                            train_size=0.3,
                            val_size=0.3,
                            test_size=0.4,
                            overlap_fraction=0.1,
                            random_state=42):
    """
    Split data sequentially by energy ranges with full overlap between train/val 
    and minor overlap with test set (highest energies).
    
    Args:
        df: DataFrame with all data
        energy_col: Name of energy column
        train_size, val_size, test_size: Size proportions (should sum to 1.0)
        overlap_fraction: Fraction of overlap between adjacent splits (e.g., 0.1 = 10%)
        random_state: For reproducibility within each energy range
        
    Returns:
        train_df, val_df, test_df: DataFrames with specified overlap pattern
    """
    # Sort by energy to ensure proper sequential splitting
    df_sorted = df.sort_values(energy_col).reset_index(drop=True)
    
    # Calculate target sample sizes
    n_total = len(df_sorted)
    n_train = int(train_size * n_total)
    n_val = int(val_size * n_total)
    n_test = int(test_size * n_total)
    
    # Calculate overlap size between train/val and test
    overlap_with_test = int(overlap_fraction * min(n_train + n_val, n_test))
    
    # Define energy ranges:
    # Test: highest energy values (end of sorted data)
    test_start = n_total - n_test
    test_end = n_total
    
    # Train/Val: lower energy values with minor overlap into test range
    trainval_start = 0
    trainval_end = test_start + overlap_with_test
    
    # Ensure boundaries are within valid range
    trainval_end = min(trainval_end, n_total)
    test_start = max(test_start, 0)
    
    # Extract the candidate pools
    # Train and Val share the same candidate pool (full overlap)
    trainval_candidates = df_sorted.iloc[trainval_start:trainval_end].copy()
    test_candidates = df_sorted.iloc[test_start:test_end].copy()
    
    # Sample to get exact target sizes
    np.random.seed(random_state)
    
    # Sample train and val from the same pool (creates full overlap potential)
    train_df = trainval_candidates.sample(n=min(n_train, len(trainval_candidates)),
                                         random_state=random_state).reset_index(drop=True)
    val_df = trainval_candidates.sample(n=min(n_val, len(trainval_candidates)),
                                       random_state=random_state + 1).reset_index(drop=True)
    test_df = test_candidates.sample(n=min(n_test, len(test_candidates)),
                                    random_state=random_state + 2).reset_index(drop=True)
    
    print(f"Sample sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Calculate energy ranges and statistics
    train_energy_range = (train_df[energy_col].min(), train_df[energy_col].max())
    val_energy_range = (val_df[energy_col].min(), val_df[energy_col].max())
    test_energy_range = (test_df[energy_col].min(), test_df[energy_col].max())
    
    print(f"\nEnergy stats:")
    print(f"  Train:")
    print(f"    - Range: {train_energy_range[0]:.4f} - {train_energy_range[1]:.4f}")
    print(f"    - Mean: {train_df[energy_col].mean():.4f}")
    print(f"  Val:")
    print(f"    - Range: {val_energy_range[0]:.4f} - {val_energy_range[1]:.4f}")
    print(f"    - Mean: {val_df[energy_col].mean():.4f}")
    print(f"  Test:")
    print(f"    - Range: {test_energy_range[0]:.4f} - {test_energy_range[1]:.4f}")
    print(f"    - Mean: {test_df[energy_col].mean():.4f}")
    
    # Calculate actual overlap statistics
    trainval_energy_min = min(train_energy_range[0], val_energy_range[0])
    trainval_energy_max = max(train_energy_range[1], val_energy_range[1])
    
    # Overlap between train/val and test
    overlap_start = max(trainval_energy_min, test_energy_range[0])
    overlap_end = min(trainval_energy_max, test_energy_range[1])
    
    if overlap_start < overlap_end:
        print(f"\nOverlap with test: {overlap_start:.4f} - {overlap_end:.4f}")
    else:
        print(f"\nNo energy overlap with test set")
        
    print(f"Train/Val share same energy range (full overlap potential)")
    
    return train_df, val_df, test_df


# ===== Model
class CO2QuantumClassifier(nn.Module):
    def __init__(self, input_dim, output_dims, p_dropout=0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(p_dropout)
        )
        
        # Separate classification heads for each Herzberg quantum number
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Dropout(p_dropout),
                nn.Linear(32, out_dim)  # raw logits
            )
            for out_dim in output_dims
        ])

    def forward(self, x):
        x = self.shared(x)
        return [head(x) for head in self.heads]


# ===== Loss functions with l2 constraint penalty
def compute_loss(outputs, targets, criterion_list, target_cols, penalty_weight=1.0):
    """
    Compute loss with optional quantum number constraints.
    Flexible to work with different target column arrangements.
    """
    total_loss = 0
    batch_size = targets.shape[0]

    # Main CE loss
    for i in range(len(outputs)):
        total_loss += criterion_list[i](outputs[i], targets[:, i])

    # Apply constraint on l2 based on v2 (only if both are present)
    v2_idx = None
    l2_idx = None
    
    # Find indices for v2 and l2 in target columns
    for i, col in enumerate(target_cols):
        if 'v2' in col.lower() or 'm2' in col.lower():
            v2_idx = i
        elif 'l2' in col.lower():
            l2_idx = i
    
    # Only apply constraint if both v2 and l2 are present
    if v2_idx is not None and l2_idx is not None and penalty_weight > 0:
        v2_pred = torch.argmax(outputs[v2_idx], dim=1)    # predicted v2
        l2_pred = torch.argmax(outputs[l2_idx], dim=1)    # predicted l2

        # Valid l2s for each v2 (quantum mechanics constraint)
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
def train(model, dataloader, optimizer, criterion_list, 
          TARGET_COLS, device, confidence_threshold=0.8):
    """
    Training function that tracks confidence statistics
    """
    model.train()
    total_loss = 0
    confidence_stats = {target: [] for target in TARGET_COLS}
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        
        # Calculate loss (can use confidence_weighted_loss if desired)
        loss = compute_loss(outputs, y, criterion_list, TARGET_COLS)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Track confidence during training
        with torch.no_grad():
            for i, target in enumerate(TARGET_COLS):
                probs = F.softmax(outputs[i], dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                confidence_stats[target].extend(max_probs.cpu().numpy())
    
    # Calculate mean confidence for this epoch
    epoch_confidence = {target: np.mean(scores) for target, scores in confidence_stats.items()}
    
    return total_loss / len(dataloader), epoch_confidence

def evaluate(model, dataloader, criterion_list, TARGET_COLS, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = compute_loss(outputs, y, criterion_list, TARGET_COLS)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# ===== Outputs
def get_predictions(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    confidences = []
    entropies = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X)
            
            # Get probabilities for each target
            probs = [F.softmax(out, dim=1) for out in outputs]
            
            # Get predictions
            preds = [torch.argmax(out, dim=1).cpu().numpy() for out in outputs]
            
            # Calculate confidence metrics for each target
            batch_confidences = []
            batch_entropies = []
            
            for prob in probs:
                # Max probability as confidence
                max_probs = torch.max(prob, dim=1)[0].cpu().numpy()
                batch_confidences.append(max_probs)
                
                # Entropy as uncertainty measure (lower = more confident)
                entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1).cpu().numpy()
                batch_entropies.append(entropy)
            
            y_pred.append(np.stack(preds, axis=1))
            y_true.append(y.numpy())
            confidences.append(np.stack(batch_confidences, axis=1))
            entropies.append(np.stack(batch_entropies, axis=1))
    
    return (np.vstack(y_true), np.vstack(y_pred), 
            np.vstack(confidences), np.vstack(entropies))


def get_mc_dropout_predictions(model, dataloader, device, n_samples=50):
    """
    Performs predictions using MC Dropout to get uncertainty estimates.
    """
    # Activate dropout layers for uncertainty estimation
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

    y_true_list = []
    # Create a list of lists to store probabilities for each head separately
    all_head_probs = [[] for _ in model.heads]

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y_true_list.append(y.numpy())
            
            # Store probabilities for each MC sample for this batch, separated by head
            batch_probs_samples_by_head = [[] for _ in model.heads]
            for _ in range(n_samples):
                outputs = model(X)
                probs = [F.softmax(out, dim=1) for out in outputs]
                
                # Append the probability tensor for each head to its corresponding list
                for i, p in enumerate(probs):
                    batch_probs_samples_by_head[i].append(p.cpu().unsqueeze(0))

            # For each head, stack the MC samples for the current batch
            for i in range(len(model.heads)):
                # Stacks along the new dimension 0, shape becomes: (n_samples, batch_size, n_classes)
                stacked_batch_probs = torch.cat(batch_probs_samples_by_head[i], dim=0)
                all_head_probs[i].append(stacked_batch_probs)

    # Concatenate results from all batches for each head
    y_true = np.vstack(y_true_list)
    
    # Each item in this list is a tensor of shape: (n_samples, total_samples, n_classes_for_head)
    final_probs_by_head = [torch.cat(head_probs, dim=1) for head_probs in all_head_probs]

    # --- Calculate final predictions and uncertainties for each head ---
    y_pred_list = []
    uncertainty_list = []

    for head_probs in final_probs_by_head:
        # 1. Mean Predictions
        mean_probs = torch.mean(head_probs, dim=0) # Shape: (total_samples, n_classes)
        y_pred_list.append(torch.argmax(mean_probs, dim=1))

        # 2. Uncertainty (Predictive Variance)
        predictive_variance = torch.var(head_probs, dim=0) # Shape: (total_samples, n_classes)
        uncertainty_list.append(torch.sum(predictive_variance, dim=1))
    
    # Stack the final 1D tensors for predictions and uncertainties
    y_pred = torch.stack(y_pred_list, dim=1).numpy()
    uncertainty = torch.stack(uncertainty_list, dim=1).numpy()

    # Restore model to standard evaluation mode
    model.eval() 
    
    return y_true, y_pred, uncertainty


def get_uncertain_predictions(y_true, y_pred, confidences, entropies, 
                              target_cols, confidence_threshold=0.5):
    """
    Identify predictions with low confidence for further analysis
    """
    uncertain_samples = []
    
    for i, target in enumerate(target_cols):
        low_conf_mask = confidences[:, i] < confidence_threshold
        
        if np.sum(low_conf_mask) > 0:
            uncertain_data = {
                'target': target,
                'sample_indices': np.where(low_conf_mask)[0],
                'true_values': y_true[low_conf_mask, i],
                'predicted_values': y_pred[low_conf_mask, i],
                'confidence_scores': confidences[low_conf_mask, i],
                'entropy_scores': entropies[low_conf_mask, i],
                'correct_predictions': (y_true[low_conf_mask, i] == y_pred[low_conf_mask, i])
            }
            uncertain_samples.append(uncertain_data)
    
    return uncertain_samples


def get_feature_importance(model, dataloader, criterion_list, device, feature_cols, target_cols, output_dir=None, seed=42):
    """
    Calculate feature importance using permutation importance.
    Uses entire dataset and random seeding.
    """
    model.eval()
    
    # Get all data from dataloader (not just first batch)
    all_X = []
    all_y = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            all_X.append(X_batch)
            all_y.append(y_batch)
    
    X_full = torch.cat(all_X, dim=0).to(device)
    y_full = torch.cat(all_y, dim=0).to(device)
    
    # Calculate baseline accuracy for each target using full dataset
    with torch.no_grad():
        outputs_baseline = model(X_full)
        baseline_preds = [torch.argmax(out, dim=1).cpu().numpy() for out in outputs_baseline]
    
    baseline_accuracies = {}
    for i, label in enumerate(target_cols):
        baseline_accuracies[label] = accuracy_score(y_full[:, i].cpu().numpy(), baseline_preds[i])

    importance_results = {}
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    for target_idx, target_label in enumerate(target_cols):
        print(f"  {target_label}...")
        feature_importance = {}
        
        with torch.no_grad():
            for feature_idx, feature_name in enumerate(feature_cols):
                # Create a copy of the full dataset
                X_permuted = X_full.clone()
                
                # Permute the values of one feature column across entire dataset
                perm_idx = torch.randperm(X_permuted.size(0), device=device)
                X_permuted[:, feature_idx] = X_permuted[perm_idx, feature_idx]

                # Get predictions with permuted data
                outputs_permuted = model(X_permuted)
                
                # Calculate accuracy for the permuted data for the current target
                permuted_preds = torch.argmax(outputs_permuted[target_idx], dim=1).cpu().numpy()
                permuted_accuracy = accuracy_score(y_full[:, target_idx].cpu().numpy(), permuted_preds)
                
                # Calculate the drop in accuracy
                importance = baseline_accuracies[target_label] - permuted_accuracy
                feature_importance[feature_name] = importance
                
        importance_results[target_label] = feature_importance

    df = pd.DataFrame(importance_results)

    if output_dir:
        plot_feature_importance(df, output_dir)

    return df