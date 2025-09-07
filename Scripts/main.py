import numpy as np
import os
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from analysis import *
from model_utils import *
from plotting import *


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
model = CO2QuantumClassifier(len(FEATURE_COLS), target_dims, p_dropout=0.1).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion_list = [nn.CrossEntropyLoss() for _ in range(len(TARGET_COLS))]

print(f"\nModel created with {len(FEATURE_COLS)} input features and {len(TARGET_COLS)} target heads")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# === Train with confidence tracking
print(f"\n{'='*60}")
print("MODEL TRAINING:")
print(f"{'='*60}")
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

print(f"\n{'='*60}")
print("MODEL PERFORMANCE:")
print(f"{'='*60}")

# === Enhanced Evaluation with Confidence
print("\nEvaluating model with confidence analysis...")
test_loss = evaluate(model, test_loader, criterion_list, TARGET_COLS, DEVICE)
print(f"  Test Loss: {test_loss:.4f}")

# === Analyze Energy Performance
print("\nAnalyzing energy performance...")
test_analysis = analyze_energy_performance(model, test_loader, test_df, TARGET_COLS, energy_col='E_original', device=DEVICE)

# Save model
torch.save(model.state_dict(), "Models/co2_model.pt")
print("  Model saved to Models/co2_model.pt")

# === Get predictions, uncertainties & confidence scores
print("\nTest set predictions and uncertainty scores with MC Dropout...")
y_true, y_pred, uncertainties = get_mc_dropout_predictions(model, test_loader, DEVICE, n_samples=50)

print("Plotting uncertainty...")
threshold = 0.15
plot_mc_dropout_uncertainty_with_correctness(y_true, y_pred, uncertainties, test_df['E_original'], train_df, TARGET_COLS, OUTPUT_DIR, threshold)
plot_acceptance_correctness_bars(y_true, y_pred, uncertainties, TARGET_COLS, OUTPUT_DIR, threshold)

# Get confidence scores and entropies
print("\nCalculating confidence scores and entropies...")
y_true, y_pred, confidences, entropies = get_predictions(model, test_loader, DEVICE)

# Standard accuracy report
print("Test Set Accuracy Results:")
results = analyze_isotopologue_predictions(y_true, y_pred, confidences, entropies, test_df, TARGET_COLS)
accuracy_df = save_accuracy_report(results, TARGET_COLS, OUTPUT_DIR)
plot_isotopologue_accuracies(results, TARGET_COLS, OUTPUT_DIR)
plot_isotopologue_comparison(results, TARGET_COLS, OUTPUT_DIR)

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
energy_values = test_df['E_original'].values
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
plot_enhanced_training_analysis(train_losses, val_losses, train_confidences, val_confidences,
                                    TARGET_COLS, OUTPUT_DIR, confidence_df)

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