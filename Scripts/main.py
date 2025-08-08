import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from model_utils import *
import os
import time


start_time = time.time()
print("Time start: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# === Config
DATA_PATH = 'Data/CO2_all_ma.txt'
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = "Data/Outputs"

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
# Capture the target_mappers
train_df, val_df, test_df, scaler, target_mappers = load_data(DATA_PATH, FEATURE_COLS, TARGET_COLS)

# Since LabelEncoder creates 0-indexed classes, we can get dimensions from max values
target_dims = []
for col in TARGET_COLS:
    max_val = max(train_df[col].max(), val_df[col].max(), test_df[col].max())
    target_dims.append(int(max_val) + 1)  # +1 because classes are 0-indexed

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


# === Model
model = CO2QuantumClassifier(len(FEATURE_COLS), target_dims).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion_list = [nn.CrossEntropyLoss() for _ in range(len(TARGET_COLS))]  # Dynamic number of loss functions

print(f"\nModel created with {len(FEATURE_COLS)} input features and {len(TARGET_COLS)} target heads")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# === Train
print("\nStarting training...")
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion_list, TARGET_COLS, DEVICE)
    val_loss = evaluate(model, val_loader, criterion_list, TARGET_COLS, DEVICE)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


# === Evaluate
print("\nEvaluating model...")
test_loss = evaluate(model, test_loader, criterion_list, TARGET_COLS, DEVICE)
print(f"Test Loss: {test_loss:.4f}")

# Save model
torch.save(model.state_dict(), "Models/co2_model.pt")
print("Model saved to Models/co2_model.pt")

# === Evaluate and Save Metrics
print("\nGenerating predictions and metrics...")
y_true, y_pred = get_predictions(model, test_loader, DEVICE)

print("\nTest Set Accuracy Results:")
save_accuracy_report(y_true, y_pred, TARGET_COLS, OUTPUT_DIR)

print("\nGenerating confusion matrices...")
plot_confusion_matrices(y_true, y_pred, TARGET_COLS, OUTPUT_DIR)

print("Plotting training curves...")
plot_loss(train_losses, val_losses, OUTPUT_DIR)

print("Calculating feature importance...")
feature_importance_df = get_feature_importance(model, test_loader, criterion_list, DEVICE, FEATURE_COLS, TARGET_COLS, OUTPUT_DIR)
print("\nFeature Importance Summary:")
print(feature_importance_df)
feature_importance_df.to_csv(os.path.join(OUTPUT_DIR, "CSVs/feature_importance.csv"))


# === Timing
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal training time: {elapsed_time:.2f} seconds")
print(f"Time per epoch: {elapsed_time/EPOCHS:.2f} seconds")

print("\nAll outputs saved to:", OUTPUT_DIR)
print("- Accuracy report: CSVs/test_accuracy.csv")
print("- Feature importance: CSVs/feature_importance.csv")
print("- Confusion matrices: Plots/confusion_*.png")
print("- Training curves: Plots/loss_plot.png")
print("- Feature importance plots: Plots/feature_importance_plots.png")