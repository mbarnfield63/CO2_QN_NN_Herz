import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import joblib  # for saving scaler & mappers

from infer_utils import load_train_data
from model_utils import CO2Dataset, CO2QuantumClassifier


def train_full_model(
    train_data_path,
    feature_cols,
    target_cols,
    p_dropout=0.1,
    batch_size=512,
    epochs=100,
    learning_rate=1e-3,
    device='cuda'
    ):
    """
    Train model on full dataset without validation split and save for inference.
    Also plots calibration curve after training.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL WITH p_dropout = {p_dropout}")
    print(f"{'='*60}")

    # Load full dataset (train only)
    train_df, scaler, target_mappers = load_train_data(
        train_data_path, feature_cols, target_cols
    )

    # Get target dimensions
    target_dims = [int(train_df[col].max()) + 1 for col in target_cols]
    print(f"Target dimensions: {target_dims}")

    # Dataset & loader
    train_ds = CO2Dataset(train_df, feature_cols, target_cols)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Model
    model = CO2QuantumClassifier(len(feature_cols), target_dims, p_dropout=p_dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_list = [nn.CrossEntropyLoss() for _ in range(len(target_cols))]

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = sum(criterion_list[i](outputs[i], y[:, i]) for i in range(len(outputs)))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}")

    print("Training completed.")

    # Plot calibration curve using the training data
    plot_calibration_curve(
        model, 
        DataLoader(train_ds, batch_size=512, shuffle=False), 
        feature_cols, 
        target_cols, 
        device,
        output_dir='Data/Outputs/', 
        p_dropout=p_dropout
    )

    return model, scaler, target_mappers


def save_model(model, scaler, target_mappers, output_dir, p_dropout):
    """
    Save model, scaler, and target mappers for later inference.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"co2_model_p{p_dropout}.pt")
    scaler_path = os.path.join(output_dir, f"scaler_p{p_dropout}.joblib")
    mappers_path = os.path.join(output_dir, f"target_mappers_p{p_dropout}.joblib")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(target_mappers, mappers_path)

    print(f"Saved model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved target mappers to: {mappers_path}")


def plot_calibration_curve(model, dataloader, feature_cols, target_cols, device, output_dir, p_dropout):
    """
    Plot a calibration curve (reliability diagram) for each target head.
    """
    print("\nGenerating calibration curve...")

    model.eval()
    confidences = []
    correctness = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            probs = [torch.softmax(out, dim=1) for out in outputs]
            preds = [torch.argmax(out, dim=1) for out in outputs]

            for head_idx, prob in enumerate(probs):
                max_conf, _ = torch.max(prob, dim=1)
                correct = (preds[head_idx] == y[:, head_idx])
                confidences.append(max_conf.cpu().numpy())
                correctness.append(correct.cpu().numpy())

    confidences = np.concatenate(confidences)
    correctness = np.concatenate(correctness)

    bins = np.linspace(0, 1, 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    acc_per_bin = []
    conf_per_bin = []

    for i in range(len(bins)-1):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if mask.any():
            acc_per_bin.append(np.mean(correctness[mask]))
            conf_per_bin.append(np.mean(confidences[mask]))
        else:
            acc_per_bin.append(0)
            conf_per_bin.append(0)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(conf_per_bin, acc_per_bin, marker="o", label="Model")

    plt.xlabel("Predicted Confidence")
    plt.ylabel("Actual Accuracy")
    plt.title(f"Calibration Curve (p_dropout={p_dropout})")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.join(output_dir, "Plots", "Calibration"), exist_ok=True)
    plot_path = os.path.join(output_dir, "Plots", "Calibration", f"calibration_p{p_dropout}.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Calibration plot saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train full CO2 model with configurable dropout.")
    parser.add_argument("--p_dropout", type=float, default=0.1,
                        help="Dropout probability (e.g., 0.1, 0.2, 0.3, 0.5)")
    parser.add_argument("--data_path", type=str, default="Data/CO2_all_ma.txt",
                        help="Path to training dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="Models",
                        help="Directory to save trained models")
    args = parser.parse_args()

    FEATURE_COLS = [
        "E", "gtot", "J", "Trove_v1", "Trove_v2", "Trove_v3", "Trove_coeff",
        "mu1", "mu2", "mu3", "mu_all", "mu1_ratio", "mu2_ratio", "mu3_ratio", "mu_all_ratio",
        "mass_c_12.0", "mass_c_13.003355",
        "mass_o_1_15.994915", "mass_o_1_16.999132", "mass_o_1_17.999161",
        "mass_o_2_15.994915", "mass_o_2_16.999132", "mass_o_2_17.999161",
        "e", "f", "Sym_A1", "Sym_A2", "Sym_B1", "Sym_B2"
    ]

    TARGET_COLS = ["hzb_v1", "hzb_v2", "hzb_l2", "hzb_v3"]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {DEVICE}")

    start_time = time.time()

    model, scaler, target_mappers = train_full_model(
        train_data_path=args.data_path,
        feature_cols=FEATURE_COLS,
        target_cols=TARGET_COLS,
        p_dropout=args.p_dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=DEVICE
    )

    save_model(model, scaler, target_mappers, args.output_dir, args.p_dropout)

    elapsed_time = time.time() - start_time
    print(f"\nTotal training time: {elapsed_time/60:.2f} minutes")
