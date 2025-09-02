import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from model_utils import load_data, CO2Dataset, CO2QuantumClassifier, get_mc_dropout_predictions
from infer_utils import decode_predictions, apply_confidence_filtering


def load_inference_data(data_path, feature_cols, target_cols, energy_cutoff=None):
    """
    Load inference dataset - same format as training data
    """
    print(f"Loading inference data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded inference dataset with {len(df)} samples and {len(df.columns)} columns.")
    
    # Drop lines with NaN values in feature columns
    df = df.dropna(subset=feature_cols)
    print(f"Dataset after dropping NaNs in features: {len(df)} samples.")

    # Apply energy cutoff if specified
    if energy_cutoff is not None:
         initial_len = len(df)
         df = df[df['E'] <= energy_cutoff]
         print(f"Applied energy cutoff of {energy_cutoff}. Samples reduced from {initial_len} to {len(df)}.")
    
    # Validation checks
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' not found in the dataset.")
    
    if df[feature_cols].isnull().any().any():
        raise ValueError("NaN values found in feature columns.")
    
    # Store original columns we need for output
    output_cols = ['iso', 'E', 'J']
    for col in output_cols:
        if col not in df.columns:
            raise ValueError(f"Required output column '{col}' not found in the dataset.")
    
    return df


def load_train_data(path,
        feature_cols,
        target_cols,
        energy_col='E',
        train_size=0.8,
        test_size=0.2,
        random_state=42,
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

    # Split the dataset into train, validation, and test sets    
    train_df, test_df = train_test_split(df, 
                                        train_size=train_size,
                                        test_size=test_size, 
                                        stratify=df[target_cols[0]],
                                        random_state=random_state)
    
    print(f"\nFinal split sizes: Train={len(train_df)}, Test={len(test_df)}")
    
    train_df.drop(columns=['iso'], inplace=True)
    test_df.drop(columns=['iso'], inplace=True)

    # Scale features (but keep original energy for plotting)
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Create target mappers for compatibility
    target_mappers = {}
    for col in target_cols:
        le = label_encoders[col]
        target_mappers[col] = {original: encoded for encoded, original in enumerate(le.classes_)}

    return train_df, test_df, scaler, target_mappers


def retrain_full_model(train_data_path, feature_cols, target_cols, 
                      batch_size=512, epochs=100, learning_rate=1e-3, device='cuda'):
    """
    Retrain model on full training dataset (no validation split for inference)
    """
    print(f"\n{'='*60}")
    print("RETRAINING MODEL ON FULL DATASET:")
    print(f"{'='*60}")
    
    # Load full training data without splitting
    train_df, _, scaler, target_mappers = load_train_data(
        train_data_path, feature_cols, target_cols
    )

    
    # Get target dimensions
    target_dims = []
    for col in target_cols:
        max_val = train_df[col].max()
        target_dims.append(int(max_val) + 1)
    
    print(f"Target dimensions: {target_dims}")
    
    # Create dataset and loader
    train_ds = CO2Dataset(train_df, feature_cols, target_cols)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = CO2QuantumClassifier(len(feature_cols), target_dims, p_dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_list = [nn.CrossEntropyLoss() for _ in range(len(target_cols))]
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop (simplified - no validation)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            
            # Simple loss calculation
            loss = sum(criterion_list[i](outputs[i], y[:, i]) for i in range(len(outputs)))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}")
    
    print("Model retraining completed.")
    return model, scaler, target_mappers


def run_inference(model, scaler, inference_df, feature_cols, target_cols, device, 
                 uncertainty_threshold=0.15, mc_samples=50):
    """
    Run inference on the loaded data with uncertainty estimation
    """
    print(f"\n{'='*60}")
    print("RUNNING INFERENCE:")
    print(f"{'='*60}")
    
    # Prepare inference data
    print("Preparing inference data...")
    
    # Store original columns needed for output
    iso_values = inference_df['iso'].values
    energy_values = inference_df['E'].values  
    j_values = inference_df['J'].values
    
    # Scale features using the scaler from training
    inference_features = inference_df[feature_cols].copy()
    inference_features_scaled = scaler.transform(inference_features)
    
    # Create dummy targets (zeros) for dataset compatibility
    dummy_targets = np.zeros((len(inference_df), len(target_cols)), dtype=np.int64)
    
    # Create dataset and loader
    inference_data_scaled = pd.DataFrame(inference_features_scaled, columns=feature_cols)
    for i, col in enumerate(target_cols):
        inference_data_scaled[col] = dummy_targets[:, i]
    
    inference_ds = CO2Dataset(inference_data_scaled, feature_cols, target_cols)
    inference_loader = DataLoader(inference_ds, batch_size=512, shuffle=False)
    
    print(f"Running MC Dropout inference with {mc_samples} samples...")
    
    # Get predictions with uncertainty
    y_true, y_pred, uncertainties = get_mc_dropout_predictions(
        model, inference_loader, device, n_samples=mc_samples
    )
    
    print(f"Generated predictions for {len(y_pred)} samples")
    print(f"Uncertainty threshold: {uncertainty_threshold}")
    
    # Apply confidence filtering
    filtered_predictions = apply_confidence_filtering(
        y_pred, uncertainties, uncertainty_threshold, target_cols
    )
    
    # Create results dataframe
    results_data = []
    n_filtered = 0
    
    for i in range(len(inference_df)):
        if filtered_predictions[i] is not None:
            # Predictions passed uncertainty threshold
            result = {
                'iso': iso_values[i],
                'energy': energy_values[i],
                'J': j_values[i],
                'hzb_v1': filtered_predictions[i][0],
                'hzb_v2': filtered_predictions[i][1], 
                'hzb_l2': filtered_predictions[i][2],
                'hzb_v3': filtered_predictions[i][3]
            }
        else:
            # Predictions filtered out due to high uncertainty
            result = {
                'iso': iso_values[i],
                'energy': energy_values[i],
                'J': j_values[i],
                'hzb_v1': None,
                'hzb_v2': None,
                'hzb_l2': None,
                'hzb_v3': None
            }
            n_filtered += 1
        
        results_data.append(result)
    
    results_df = pd.DataFrame(results_data)
    
    print(f"Uncertainty filtering results:")
    print(f"  Total samples: {len(results_df)}")
    print(f"  Samples with predictions: {len(results_df) - n_filtered}")
    print(f"  Samples filtered out: {n_filtered} ({n_filtered/len(results_df)*100:.1f}%)")
    
    return results_df


def save_inference_results(results_df, output_path):
    """
    Save predictions to file with iso, energy, J, and quantum numbers
    """
    print(f"\nSaving inference results to: {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Saved {len(results_df)} predictions to {output_path}")
    
    # Print summary statistics
    non_null_mask = results_df['hzb_v1'].notna()
    print(f"\nSummary:")
    print(f"  Total entries: {len(results_df)}")
    print(f"  With predictions: {non_null_mask.sum()}")
    print(f"  Without predictions (high uncertainty): {(~non_null_mask).sum()}")


def main():
    """
    Main inference pipeline
    """
    # Configuration
    TRAIN_DATA_PATH = 'Data/CO2_all_ma.txt'
    INFERENCE_DATA_PATH = 'Data/CO2_all_ca.txt'
    OUTPUT_PATH = 'Data/hzb_predictions.csv'
    
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
    ENERGY_CUTOFF = 25000  # cm^-1
    UNCERTAINTY_THRESHOLD = 0.25

    start_time = time.time()
    print("Time start: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    print("Starting inference pipeline...")
    print(f"Device: {DEVICE}")
    
    try:
        # Step 1: Load inference data
        inference_df = load_inference_data(INFERENCE_DATA_PATH, FEATURE_COLS, TARGET_COLS, ENERGY_CUTOFF)
        
        # Step 2: Retrain model on full training data
        model, scaler, target_mappers = retrain_full_model(
            TRAIN_DATA_PATH, FEATURE_COLS, TARGET_COLS, device=DEVICE
        )
        
        # Step 3: Run inference with uncertainty estimation
        results_df = run_inference(
            model, scaler, inference_df, FEATURE_COLS, TARGET_COLS, DEVICE,
            uncertainty_threshold=UNCERTAINTY_THRESHOLD
        )
        
        # Step 4: Save results
        save_inference_results(results_df, OUTPUT_PATH)
        
        print(f"\n{'='*60}")
        print("INFERENCE COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print("Time end: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        elapsed_time = time.time() - start_time
        print(f"Total time elapsed: {elapsed_time/60:.2f} minutes")
        
    except Exception as e:
        print(f"\nError during inference: {str(e)}")
        print("Exiting.")
        print("Time end: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        elapsed_time = time.time() - start_time
        raise


if __name__ == "__main__":
    main()