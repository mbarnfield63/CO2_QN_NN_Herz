import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Optional, Dict, Any


def apply_confidence_filtering(predictions: np.ndarray, 
                                uncertainties: np.ndarray, 
                                threshold: float,
                                target_cols: List[str]) -> List[Optional[List[int]]]:
    """
    Filter predictions based on uncertainty threshold.
    If ANY target has high uncertainty, filter out ALL predictions for that sample.
    
    Args:
        predictions: Array of shape (n_samples, n_targets)
        uncertainties: Array of shape (n_samples, n_targets) 
        threshold: Uncertainty threshold - predictions above this are filtered out
        target_cols: List of target column names for logging
    
    Returns:
        List of predictions where None indicates filtered sample
    """
    n_samples = len(predictions)
    filtered_predictions = []
    
    # Count filtering statistics
    filtered_by_target = {target: 0 for target in target_cols}
    total_filtered = 0
    
    for i in range(n_samples):
        sample_uncertainties = uncertainties[i, :]
        sample_predictions = predictions[i, :]
        
        # Check if ANY target exceeds uncertainty threshold
        high_uncertainty_mask = sample_uncertainties > threshold
        
        if np.any(high_uncertainty_mask):
            # Filter out this entire sample
            filtered_predictions.append(None)
            total_filtered += 1
            
            # Track which targets caused filtering
            for j, is_high_uncertainty in enumerate(high_uncertainty_mask):
                if is_high_uncertainty:
                    filtered_by_target[target_cols[j]] += 1
        else:
            # Keep predictions for this sample
            filtered_predictions.append(sample_predictions.tolist())
    
    # Print filtering statistics
    print(f"\nUncertainty filtering statistics:")
    print(f"  Threshold: {threshold}")
    print(f"  Total samples filtered: {total_filtered}/{n_samples} ({total_filtered/n_samples*100:.1f}%)")
    print(f"  Samples passing filter: {n_samples - total_filtered}/{n_samples} ({(n_samples - total_filtered)/n_samples*100:.1f}%)")
    
    print(f"\nFiltering by target (samples with high uncertainty):")
    for target, count in filtered_by_target.items():
        print(f"  {target}: {count} samples ({count/n_samples*100:.1f}%)")
    
    return filtered_predictions


def decode_predictions(encoded_predictions: np.ndarray, 
                        target_mappers: Dict[str, Dict[int, Any]]) -> Dict[str, List]:
    """
    Decode integer predictions back to original labels using target mappers.
    
    Args:
        encoded_predictions: Array of shape (n_samples, n_targets) with integer predictions
        target_mappers: Dict mapping target names to {encoded: original} mappings
        
    Returns:
        Dictionary with target names as keys and decoded predictions as values
    """
    decoded_results = {}
    target_names = list(target_mappers.keys())
    
    for i, target in enumerate(target_names):
        mapper = target_mappers[target]
        # Create reverse mapping: encoded -> original
        reverse_mapper = {encoded: original for original, encoded in mapper.items()}
        
        # Decode predictions for this target
        encoded_preds = encoded_predictions[:, i]
        decoded_preds = [reverse_mapper.get(pred, f"UNKNOWN_{pred}") for pred in encoded_preds]
        decoded_results[target] = decoded_preds
    
    return decoded_results


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
    
    df.drop(columns=['iso'], inplace=True)

    # Scale features (but keep original energy for plotting)
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Create target mappers for compatibility
    target_mappers = {}
    for col in target_cols:
        le = label_encoders[col]
        target_mappers[col] = {original: encoded for encoded, original in enumerate(le.classes_)}

    return df, scaler, target_mappers


def match_by_energy_and_isotopologue(original_data: pd.DataFrame, 
                                    inference_results: pd.DataFrame,
                                    energy_tolerance: float = 1e-6) -> pd.DataFrame:
    """
    Match inference results back to original data by isotopologue and energy.
    
    Args:
        original_data: Original dataset with isotopologue and energy info
        inference_results: Results from inference with predictions
        energy_tolerance: Tolerance for energy matching
        
    Returns:
        Merged dataframe with original data and predictions
    """
    print(f"Matching {len(inference_results)} inference results to {len(original_data)} original samples...")
    
    # Merge on isotopologue and energy (with tolerance handling if needed)
    merged = pd.merge(
        original_data, 
        inference_results, 
        on=['iso', 'energy', 'J'], 
        how='left',
        suffixes=('_original', '_predicted')
    )
    
    print(f"Matched {len(merged)} samples")
    
    # Count how many got predictions
    has_predictions = merged['hzb_v1'].notna().sum()
    print(f"  {has_predictions} samples have predictions")
    print(f"  {len(merged) - has_predictions} samples missing predictions")
    
    return merged


def replace_values_in_file(input_file_path: str, 
                            replacements: pd.DataFrame,
                            output_file_path: str,
                            key_columns: List[str] = ['iso', 'energy', 'J'],
                            value_columns: List[str] = ['hzb_v1', 'hzb_v2', 'hzb_l2', 'hzb_v3']) -> int:
    """
    Replace values in a file based on key matching.
    
    Args:
        input_file_path: Path to input file to modify
        replacements: DataFrame with keys and new values
        output_file_path: Path to save modified file
        key_columns: Columns to use for matching
        value_columns: Columns to replace
        
    Returns:
        Number of replacements made
    """
    print(f"Replacing values in {input_file_path}...")
    
    # Load original file
    original_df = pd.read_csv(input_file_path)
    print(f"  Loaded {len(original_df)} rows from original file")
    
    # Create a copy for modifications
    modified_df = original_df.copy()
    replacements_made = 0
    
    # Create a lookup dictionary from replacements
    replacement_dict = {}
    for _, row in replacements.iterrows():
        if pd.notna(row[value_columns[0]]):  # Only if we have valid predictions
            key = tuple(row[key_col] for key_col in key_columns)
            values = {col: row[col] for col in value_columns}
            replacement_dict[key] = values
    
    print(f"  {len(replacement_dict)} replacement entries available")
    
    # Apply replacements
    for idx, row in modified_df.iterrows():
        key = tuple(row[key_col] for key_col in key_columns)
        
        if key in replacement_dict:
            replacement_values = replacement_dict[key]
            for col in value_columns:
                if col in modified_df.columns:
                    modified_df.at[idx, col] = replacement_values[col]
            replacements_made += 1
    
    # Save modified file
    modified_df.to_csv(output_file_path, index=False)
    print(f"  Made {replacements_made} replacements")
    print(f"  Saved modified file to {output_file_path}")
    
    return replacements_made


def analyze_prediction_coverage(inference_results: pd.DataFrame, 
                                target_cols: List[str]) -> Dict[str, Any]:
    """
    Analyze the coverage and quality of predictions.
    
    Args:
        inference_results: DataFrame with prediction results
        target_cols: List of target column names
        
    Returns:
        Dictionary with analysis results
    """
    total_samples = len(inference_results)
    
    # Count non-null predictions for each target
    coverage_stats = {}
    for target in target_cols:
        non_null_count = inference_results[target].notna().sum()
        coverage_stats[target] = {
            'count': non_null_count,
            'percentage': (non_null_count / total_samples) * 100
        }
    
    # Overall coverage (samples with at least one prediction)
    has_any_prediction = inference_results[target_cols].notna().any(axis=1).sum()
    
    # Complete coverage (samples with all predictions)
    has_all_predictions = inference_results[target_cols].notna().all(axis=1).sum()
    
    analysis = {
        'total_samples': total_samples,
        'has_any_prediction': has_any_prediction,
        'has_all_predictions': has_all_predictions,
        'coverage_by_target': coverage_stats,
        'any_prediction_percentage': (has_any_prediction / total_samples) * 100,
        'complete_prediction_percentage': (has_all_predictions / total_samples) * 100
    }
    
    return analysis


def print_prediction_summary(analysis: Dict[str, Any], target_cols: List[str]):
    """
    Print a summary of prediction coverage analysis.
    """
    print(f"\n{'='*50}")
    print("PREDICTION COVERAGE SUMMARY")
    print(f"{'='*50}")
    
    print(f"Total samples: {analysis['total_samples']}")
    print(f"Samples with any prediction: {analysis['has_any_prediction']} ({analysis['any_prediction_percentage']:.1f}%)")
    print(f"Samples with all predictions: {analysis['has_all_predictions']} ({analysis['complete_prediction_percentage']:.1f}%)")
    
    print(f"\nCoverage by target:")
    for target in target_cols:
        stats = analysis['coverage_by_target'][target]
        print(f"  {target}: {stats['count']} ({stats['percentage']:.1f}%)")


def validate_inference_output(results_df: pd.DataFrame, 
                                expected_columns: List[str] = ['iso', 'energy', 'J', 'hzb_v1', 'hzb_v2', 'hzb_l2', 'hzb_v3']) -> bool:
    """
    Validate that inference output has expected format.
    
    Args:
        results_df: Results dataframe to validate
        expected_columns: List of expected column names
        
    Returns:
        True if valid, False otherwise
    """
    print("Validating inference output format...")
    
    # Check columns exist
    missing_columns = set(expected_columns) - set(results_df.columns)
    if missing_columns:
        print(f"  ERROR: Missing columns: {missing_columns}")
        return False
    
    # Check data types
    required_numeric = ['energy', 'J']
    for col in required_numeric:
        if not pd.api.types.is_numeric_dtype(results_df[col]):
            print(f"  ERROR: Column {col} should be numeric")
            return False
    
    # Check for reasonable values
    if results_df['energy'].min() < 0:
        print(f"  WARNING: Negative energy values found")
    
    if results_df['J'].min() < 0:
        print(f"  WARNING: Negative J values found")
    
    print("  Validation passed!")
    return True