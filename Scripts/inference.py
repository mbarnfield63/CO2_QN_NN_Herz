import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from model_utils import CO2Dataset, CO2QuantumClassifier, get_mc_dropout_predictions


def load_inference_data_chunked(data_path, feature_cols, target_cols, chunk_size=0.1, energy_cutoff=None):
    """
    Load inference dataset in chunks to manage memory
    Returns a generator that yields chunks of data
    """
    print(f"Loading inference data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded inference dataset with {len(df)} samples and {len(df.columns)} columns.")
    
    # Apply energy cutoff filter if specified
    if energy_cutoff is not None:
        original_count = len(df)
        df = df[df['E'] <= energy_cutoff]
        print(f"Applied energy cutoff ≤ {energy_cutoff}: {original_count} → {len(df)} samples")
    
    # Drop lines with NaN values in feature columns
    df = df.dropna(subset=feature_cols)
    print(f"Dataset after dropping NaNs in features: {len(df)} samples.")

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
    
    # Calculate chunk size
    total_samples = len(df)
    samples_per_chunk = int(total_samples * chunk_size)
    if samples_per_chunk == 0:
        samples_per_chunk = 1
    
    n_chunks = (total_samples + samples_per_chunk - 1) // samples_per_chunk
    print(f"Will process {total_samples} samples in {n_chunks} chunks of ~{samples_per_chunk} samples each")
    
    # Yield chunks
    for i in range(n_chunks):
        start_idx = i * samples_per_chunk
        end_idx = min((i + 1) * samples_per_chunk, total_samples)
        chunk_df = df.iloc[start_idx:end_idx].copy()
        yield i + 1, n_chunks, chunk_df


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


def run_inference_chunk(model, scaler, chunk_df, feature_cols, target_cols, device, mc_samples=50):
    """
    Run inference on a single chunk of data with uncertainty estimation
    Returns results with uncertainty values included
    """
    # Prepare inference data
    iso_values = chunk_df['iso'].values
    energy_values = chunk_df['E'].values  
    j_values = chunk_df['J'].values
    
    # Scale features using the scaler from training
    inference_features = chunk_df[feature_cols].copy()
    inference_features_scaled = scaler.transform(inference_features)
    
    # Create dummy targets (zeros) for dataset compatibility
    dummy_targets = np.zeros((len(chunk_df), len(target_cols)), dtype=np.int64)
    
    # Create dataset and loader
    inference_data_scaled = pd.DataFrame(inference_features_scaled, columns=feature_cols)
    for i, col in enumerate(target_cols):
        inference_data_scaled[col] = dummy_targets[:, i]
    
    inference_ds = CO2Dataset(inference_data_scaled, feature_cols, target_cols)
    inference_loader = DataLoader(inference_ds, batch_size=256, shuffle=False)  # Smaller batch size for chunks
    
    # Get predictions with uncertainty
    _, y_pred, uncertainties = get_mc_dropout_predictions(
        model, inference_loader, device, n_samples=mc_samples
    )
    
    # Create results for this chunk
    results_data = []
    
    for i in range(len(chunk_df)):
        # Calculate average uncertainty across all targets
        # uncertainties has shape (n_samples, n_targets), so we access it as uncertainties[i, j]
        avg_uncertainty = np.mean(uncertainties[i, :])
        
        result = {
            'iso': iso_values[i],
            'energy': energy_values[i],
            'J': j_values[i],
            'hzb_v1': y_pred[i, 0],
            'hzb_v2': y_pred[i, 1], 
            'hzb_l2': y_pred[i, 2],
            'hzb_v3': y_pred[i, 3],
            'uncertainty': avg_uncertainty,
            'uncertainty_v1': uncertainties[i, 0],
            'uncertainty_v2': uncertainties[i, 1],
            'uncertainty_l2': uncertainties[i, 2],
            'uncertainty_v3': uncertainties[i, 3]
        }
        results_data.append(result)
    
    return pd.DataFrame(results_data)


def initialize_output_file(output_path):
    """
    Initialize the output CSV file with headers
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create header row
    headers = ['iso', 'energy', 'J', 'hzb_v1', 'hzb_v2', 'hzb_l2', 'hzb_v3', 
               'uncertainty', 'uncertainty_v1', 'uncertainty_v2', 'uncertainty_l2', 'uncertainty_v3']
    
    # Write header to file
    with open(output_path, 'w') as f:
        f.write(','.join(headers) + '\n')
    
    print(f"Initialized output file: {output_path}")


def append_results_to_file(results_df, output_path):
    """
    Append results to the output CSV file
    """
    results_df.to_csv(output_path, mode='a', header=False, index=False)


def run_chunked_inference(model, scaler, inference_data_path, feature_cols, target_cols, 
                         device, output_path, energy_cutoff=None, chunk_size=0.1, mc_samples=50):
    """
    Run inference on data in chunks, appending results to output file
    """
    print(f"\n{'='*60}")
    print("RUNNING CHUNKED INFERENCE:")
    print(f"{'='*60}")
    
    if energy_cutoff is not None:
        print(f"Energy cutoff: ≤ {energy_cutoff}")
    
    # Initialize output file
    initialize_output_file(output_path)
    
    total_processed = 0
    total_chunks = 0
    
    # Process each chunk - pass energy_cutoff to the chunked loader
    for chunk_num, n_chunks, chunk_df in load_inference_data_chunked(
        inference_data_path, feature_cols, target_cols, chunk_size, energy_cutoff
    ):
        print(f"\nProcessing chunk {chunk_num}/{n_chunks} ({len(chunk_df)} samples)...")
        
        # Run inference on this chunk
        chunk_results = run_inference_chunk(
            model, scaler, chunk_df, feature_cols, target_cols, device, mc_samples
        )
        
        # Append results to file
        append_results_to_file(chunk_results, output_path)
        
        total_processed += len(chunk_df)
        total_chunks += 1
        
        print(f"Chunk {chunk_num} completed. Total processed: {total_processed}")
        
        # Clear variables to free memory
        del chunk_df, chunk_results
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\nChunked inference completed!")
    print(f"Total chunks processed: {total_chunks}")
    print(f"Total samples processed: {total_processed}")
    print(f"Results saved to: {output_path}")


def print_final_summary(output_path):
    """
    Print summary statistics of the final results
    """
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY:")
    print(f"{'='*60}")
    
    # Read the final results file to get summary stats
    try:
        results_df = pd.read_csv(output_path)
        
        print(f"Total entries in output file: {len(results_df)}")
        print(f"Columns: {list(results_df.columns)}")
        
        # Uncertainty statistics
        if 'uncertainty' in results_df.columns:
            print(f"\nUncertainty statistics:")
            print(f"  Mean uncertainty: {results_df['uncertainty'].mean():.4f}")
            print(f"  Median uncertainty: {results_df['uncertainty'].median():.4f}")
            print(f"  Min uncertainty: {results_df['uncertainty'].min():.4f}")
            print(f"  Max uncertainty: {results_df['uncertainty'].max():.4f}")
            
            # Show uncertainty distribution
            for threshold in [0.1, 0.15, 0.2, 0.25, 0.3]:
                count = (results_df['uncertainty'] <= threshold).sum()
                pct = count / len(results_df) * 100
                print(f"  Samples with uncertainty ≤ {threshold}: {count} ({pct:.1f}%)")
        
    except Exception as e:
        print(f"Could not read output file for summary: {e}")


def main():
    """
    Main inference pipeline with chunked processing
    """
    # Configuration
    TRAIN_DATA_PATH = 'Data/CO2_all_ma.txt'
    INFERENCE_DATA_PATH = 'Data/CO2_all_ca.txt'
    OUTPUT_PATH = 'Data/hzb_predictions_with_uncertainty.csv'
    
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
    CHUNK_SIZE = 0.1  # Process 10% of data at a time
    MC_SAMPLES = 100
    ENERGY_CUTOFF = 15000  # Optional energy cutoff filter

    start_time = time.time()
    print("Time start: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    print("Starting chunked inference pipeline...")
    print(f"Device: {DEVICE}")
    print(f"Chunk size: {CHUNK_SIZE*100:.0f}% of data per chunk")
    
    try:
        # Step 1: Retrain model on full training data
        model, scaler, target_mappers = retrain_full_model(
            TRAIN_DATA_PATH, FEATURE_COLS, TARGET_COLS, device=DEVICE
        )
        
        # Step 2: Run chunked inference with uncertainty estimation
        run_chunked_inference(
            model, scaler, INFERENCE_DATA_PATH, FEATURE_COLS, TARGET_COLS, 
            DEVICE, OUTPUT_PATH, energy_cutoff=ENERGY_CUTOFF, chunk_size=CHUNK_SIZE, mc_samples=MC_SAMPLES
        )
        
        # Step 3: Print final summary
        print_final_summary(OUTPUT_PATH)
        
        print(f"\n{'='*60}")
        print("CHUNKED INFERENCE COMPLETED SUCCESSFULLY!")
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