# Enhanced main.py with confidence tracking

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from model_utils import *
import os
import time
import numpy as np

def analyze_energy_performance(model, test_loader, test_df, energy_col='E_original', device='cpu'):
    """Analyze model performance across energy ranges."""
    y_true, y_pred = get_predictions(model, test_loader, device)
    
    # Add predictions to test dataframe for analysis
    test_analysis = test_df.copy()
    for i, col in enumerate(TARGET_COLS):
        test_analysis[f'{col}_pred'] = y_pred[:, i]
        test_analysis[f'{col}_correct'] = (y_true[:, i] == y_pred[:, i])
    
    # Analyze performance by energy quartiles
    energy_quartiles = pd.qcut(test_analysis[energy_col], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    test_analysis['energy_quartile'] = energy_quartiles
    
    print("\nPerformance by Energy Quartile:")
    for col in TARGET_COLS:
        accuracy_by_quartile = test_analysis.groupby('energy_quartile')[f'{col}_correct'].mean()
        print(f"{col}:")
        for quartile, acc in accuracy_by_quartile.items():
            print(f"  {quartile}: {acc:.4f}")
    
    return test_analysis


start_time = time.time()
print("Time start: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# === Config
DATA_PATH = 'Data/CO2_all_ma.txt'
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = "Data/Outputs"
CONFIDENCE_THRESHOLD = 0.8  # Threshold for confident predictions

# Create all necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "CSVs"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Plots"), exist_ok=True)
os.makedirs("Models", exist_ok=True)

# === Columns
FEATURE_COLS = [
    "E", "gtot", "J", "Trove_v1", "Trove_v2", "Trove_v3", "Trove_coeff",
    "mu1", "mu2", "mu3", "mu_all", "mu1_ratio", "mu2_ratio", "mu3_ratio", "mu_all_ratio",
    "mass_c_12.0", "mass_c_13.003355",
    "mass_o_1_15.994915", "mass_o_1_16.999132", "mass_o_1_17.999161",
    "mass_o_2_15.994915", "mass_o_2_16.999132", "mass_o_2_17.999161",
    "e", "f", "Sym_Adp", "Sym_Ap", "Sym_A1", "Sym_A2", "Sym_B1", "Sym_B2"
]

TARGET_COLS = ["hzb_v1", "hzb_v2", "hzb_l2", "hzb_v3"]

# === Data
train_df, val_df, test_df, scaler, target_mappers = load_data(DATA_PATH,
                                                              FEATURE_COLS,
                                                              TARGET_COLS,
                                                              energy_splitting=True,
                                                              output_dir=OUTPUT_DIR)

# Get target dimensions
target_dims = []
for col in TARGET_COLS:
    max_val = max(train_df[col].max(), val_df[col].max(), test_df[col].max())
    target_dims.append(int(max_val) + 1)

print(f"\nTarget dimensions (number of classes): {target_dims}")
print(f"Target mappers keys: {list(target_mappers.keys())}")

train_ds = CO2Dataset(train_df, FEATURE_COLS, TARGET_COLS)
val_ds = CO2Dataset(val_df, FEATURE_COLS, TARGET_COLS)
test_ds = CO2Dataset(test_df, FEATURE_COLS, TARGET_COLS)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# === Trackers
train_losses = []
val_losses = []
train_confidences = []  # Track confidence during training
val_confidences = []

# === Model
model = CO2QuantumClassifier(len(FEATURE_COLS), target_dims).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion_list = [nn.CrossEntropyLoss() for _ in range(len(TARGET_COLS))]

print(f"\nModel created with {len(FEATURE_COLS)} input features and {len(TARGET_COLS)} target heads")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# === Train with confidence tracking
print("\nStarting training with confidence tracking...")
for epoch in range(EPOCHS):
    # Training with confidence tracking
    train_loss, train_conf = train(
        model, train_loader, optimizer, criterion_list, TARGET_COLS, DEVICE, CONFIDENCE_THRESHOLD
    )
    
    # Validation
    val_loss = evaluate(model, val_loader, criterion_list, TARGET_COLS, DEVICE)
    
    # Get validation confidence
    _, _, val_conf_scores, _ = get_predictions(model, val_loader, DEVICE)
    val_conf = {target: np.mean(val_conf_scores[:, i]) for i, target in enumerate(TARGET_COLS)}
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_confidences.append(train_conf)
    val_confidences.append(val_conf)

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f" "*14 + f"Train Conf: {np.mean(list(train_conf.values())):.3f} | Val Conf: {np.mean(list(val_conf.values())):.3f}")

# === Enhanced Evaluation with Confidence
print("\nEvaluating model with confidence analysis...")
test_loss = evaluate(model, test_loader, criterion_list, TARGET_COLS, DEVICE)
print(f"  Test Loss: {test_loss:.4f}")

# === Analyze Energy Performance
print("\nAnalyzing energy performance...")
test_analysis = analyze_energy_performance(model, test_loader, test_df, energy_col='E_original', device=DEVICE)

# Save model
torch.save(model.state_dict(), "Models/co2_model.pt")
print("  Model saved to Models/co2_model.pt")

# === Get predictions & confidence scores
print("\nTest set predictions and confidence scores...")
y_true, y_pred, confidences, entropies = get_predictions(model, test_loader, DEVICE)

# Standard accuracy report
print("Test Set Accuracy Results:")
save_accuracy_report(y_true, y_pred, TARGET_COLS, OUTPUT_DIR)

# Confidence analysis
print("\nAnalyzing confidence scores...")
for i, target in enumerate(TARGET_COLS):
    mean_conf = np.mean(confidences[:, i])
    mean_entropy = np.mean(entropies[:, i])
    
    # Accuracy for high vs low confidence predictions
    high_conf_mask = confidences[:, i] >= CONFIDENCE_THRESHOLD
    low_conf_mask = confidences[:, i] < CONFIDENCE_THRESHOLD
    
    if np.sum(high_conf_mask) > 0:
        high_conf_acc = accuracy_score(y_true[high_conf_mask, i], y_pred[high_conf_mask, i])
    else:
        high_conf_acc = 0.0
        
    if np.sum(low_conf_mask) > 0:
        low_conf_acc = accuracy_score(y_true[low_conf_mask, i], y_pred[low_conf_mask, i])
    else:
        low_conf_acc = 0.0
    
    print(f"{target}:")
    print(f"  Mean Confidence: {mean_conf:.3f}, Mean Entropy: {mean_entropy:.3f}")
    print(f"  High Conf Acc ({np.sum(high_conf_mask)} samples): {high_conf_acc:.3f}")
    print(f"  Low Conf Acc ({np.sum(low_conf_mask)} samples): {low_conf_acc:.3f}")

# === Energy-based confidence analysis
print("\nAnalyzing confidence by energy regions...")
energy_values = test_df['E'].values
confidence_by_energy(
    y_true, y_pred, confidences, entropies, energy_values, TARGET_COLS, OUTPUT_DIR
)

# Plot confidence distributions
print("Plotting confidence distributions...")
plot_confidence_distribution(confidences, entropies, TARGET_COLS, OUTPUT_DIR)

# === Find uncertain predictions for analysis
print("\nIdentifying uncertain predictions...")
uncertain_samples = get_uncertain_predictions(
    y_true, y_pred, confidences, entropies, TARGET_COLS, confidence_threshold=0.3
)

for uncertain_data in uncertain_samples:
    target = uncertain_data['target']
    n_uncertain = len(uncertain_data['sample_indices'])
    correct_uncertain = np.sum(uncertain_data['correct_predictions'])
    
    print(f"{target}: {n_uncertain} uncertain predictions ({correct_uncertain} correct)")

# === Save detailed confidence results
confidence_results = []
for i in range(len(y_true)):
    for j, target in enumerate(TARGET_COLS):
        confidence_results.append({
            'sample_idx': i,
            'target': target,
            'true_value': y_true[i, j],
            'predicted_value': y_pred[i, j],
            'confidence': confidences[i, j],
            'entropy': entropies[i, j],
            'correct': y_true[i, j] == y_pred[i, j],
            'energy': energy_values[i] if i < len(energy_values) else None
        })

confidence_df = pd.DataFrame(confidence_results)
confidence_df.to_csv(os.path.join(OUTPUT_DIR, "CSVs/detailed_confidence_results.csv"), index=False)

# === Plot training curves with confidence
print("Plotting enhanced training curves...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Loss curves
ax1.plot(train_losses, label="Train Loss")
ax1.plot(val_losses, label="Val Loss")
ax1.set_title("Training & Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

# Overall confidence curves
train_conf_means = [np.mean(list(conf.values())) for conf in train_confidences]
val_conf_means = [np.mean(list(conf.values())) for conf in val_confidences]

ax2.plot(train_conf_means, label="Train Confidence")
ax2.plot(val_conf_means, label="Val Confidence")
ax2.set_title("Mean Confidence During Training")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Mean Confidence")
ax2.legend()

# Individual target confidence (validation)
for i, target in enumerate(TARGET_COLS):
    target_confs = [conf[target] for conf in val_confidences]
    ax3.plot(target_confs, label=target)
ax3.set_title("Validation Confidence by Target")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Confidence")
ax3.legend()

# Confidence vs Accuracy scatter
for i, target in enumerate(TARGET_COLS):
    target_mask = confidence_df['target'] == target
    target_data = confidence_df[target_mask]
    
    # Group by confidence bins for cleaner visualization
    conf_bins = np.linspace(0, 1, 20)
    conf_binned = pd.cut(target_data['confidence'], bins=conf_bins)
    binned_acc = target_data.groupby(conf_binned)['correct'].mean()
    bin_centers = [(conf_bins[i] + conf_bins[i+1])/2 for i in range(len(conf_bins)-1)]
    
    ax4.plot(bin_centers, binned_acc.values, 'o-', label=target, alpha=0.7)

ax4.set_title("Confidence vs Accuracy")
ax4.set_xlabel("Confidence Score")
ax4.set_ylabel("Accuracy")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Plots/enhanced_training_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

# === Generate other standard outputs
print("\nGenerating confusion matrices...")
plot_confusion_matrices(y_true, y_pred, TARGET_COLS, OUTPUT_DIR)

print("\nPlotting loss curves...")
plot_loss(train_losses, val_losses, OUTPUT_DIR)

print("\nCalculating feature importance...")
feature_importance_df = get_feature_importance(model, test_loader, criterion_list, DEVICE, FEATURE_COLS, TARGET_COLS, OUTPUT_DIR)
feature_importance_df.to_csv(os.path.join(OUTPUT_DIR, "CSVs/feature_importance.csv"))

# === Final Summary
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal training time: {elapsed_time:.2f} seconds")
print(f"  Time per epoch: {elapsed_time/EPOCHS:.2f} seconds")

print("\nAll outputs saved to:", OUTPUT_DIR)