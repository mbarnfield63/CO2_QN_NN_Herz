import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib.gridspec import GridSpec

def plot_energy_distributions_detailed(train_df, val_df, test_df, 
                                       energy_col='E_original', 
                                       output_dir=None):
    """
    Plot energy distributions across splits showing both original and scaled values.
    """
    plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, height_ratios=[2, 1])

    # Plot 1: Energy Distribution (Top Left)
    ax1 = plt.subplot(gs[0, 0])
    for data, label, color in zip([train_df[energy_col], val_df[energy_col], test_df[energy_col]], 
                                  ['Train', 'Val', 'Test'],
                                  ['blue', 'orange', 'green']):
        ax1.hist(data, bins=50, alpha=0.3, density=True, label=label, color=color)
    ax1.set_xlabel('Energy (Original Units)')
    ax1.set_ylabel('Density')
    ax1.set_title('Energy Distribution Overlap')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Energy Box Plot (Top Right)
    ax2 = plt.subplot(gs[0, 1])
    data_to_plot = [train_df[energy_col], val_df[energy_col], test_df[energy_col]]
    box_plot = ax2.boxplot(data_to_plot, labels=['Train', 'Val', 'Test'], patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel('Energy (Original Units)')
    ax2.set_title('Original Energy Box Plot')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Statistics Table (Bottom, spanning both columns)
    ax3 = plt.subplot(gs[1, :])
    ax3.axis('tight')
    ax3.axis('off')

    stats_data = {
        'Split': ['Train', 'Val', 'Test'],
        'Count': [len(train_df), len(val_df), len(test_df)],
        'Mean': [train_df[energy_col].mean(), val_df[energy_col].mean(), test_df[energy_col].mean()],
        'Median': [train_df[energy_col].median(), val_df[energy_col].median(), test_df[energy_col].median()],
        'Std': [train_df[energy_col].std(), val_df[energy_col].std(), test_df[energy_col].std()],
        'Min': [train_df[energy_col].min(), val_df[energy_col].min(), test_df[energy_col].min()],
        'Max': [train_df[energy_col].max(), val_df[energy_col].max(), test_df[energy_col].max()]
    }

    table_data = []
    for i in range(3):
        row = [
            stats_data['Split'][i],
            f"{stats_data['Count'][i]}",
            f"{stats_data['Mean'][i]:.2f}",
            f"{stats_data['Median'][i]:.2f}",
            f"{stats_data['Std'][i]:.2f}",
            f"{stats_data['Min'][i]:.2f}",
            f"{stats_data['Max'][i]:.2f}"
        ]
        table_data.append(row)

    table = ax3.table(cellText=table_data,
                      colLabels=['Split', 'Count', 'Mean', 'Median', 'Std', 'Min', 'Max'],
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    ax3.set_title('Energy Statistics (Original Units)')
        
    plt.tight_layout()
   
    if output_dir:
        os.makedirs(os.path.join(output_dir, "Plots"), exist_ok=True)
        plt.savefig(os.path.join(output_dir, "Plots/detailed_energy_distribution_analysis.png"), 
                   dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed statistics
    print(f"\n{'='*60}")
    print("DETAILED ENERGY STATISTICS (Original Units)")
    print(f"{'='*60}")
    
    for split_name, split_df in [('TRAIN', train_df), ('VALIDATION', val_df), ('TEST', test_df)]:
        energy_data = split_df[energy_col]
        print(f"\n{split_name} SET:")
        print(f"  Count:      {len(energy_data):>8,}")
        print(f"  Mean:       {energy_data.mean():>8.4f}")
        print(f"  Median:     {energy_data.median():>8.4f}")
        print(f"  Std:        {energy_data.std():>8.4f}")
        print(f"  Min:        {energy_data.min():>8.4f}")
        print(f"  Max:        {energy_data.max():>8.4f}")
        print(f"  Range:      {energy_data.max() - energy_data.min():>8.4f}")
        print(f"  P10:        {np.percentile(energy_data, 10):>8.4f}")
        print(f"  P25:        {np.percentile(energy_data, 25):>8.4f}")
        print(f"  P75:        {np.percentile(energy_data, 75):>8.4f}")
        print(f"  P90:        {np.percentile(energy_data, 90):>8.4f}")
    
    print(f"\n{'='*60}")
    print("ENERGY BIAS ANALYSIS:")
    print(f"{'='*60}")
    print(f"Train mean energy: {train_df[energy_col].mean():.4f}")
    print(f"Test mean energy:  {test_df[energy_col].mean():.4f}")
    print(f"Energy bias ratio: {test_df[energy_col].mean() / train_df[energy_col].mean():.4f}")
    
    # Calculate overlap
    train_range = (train_df[energy_col].min(), train_df[energy_col].max())
    test_range = (test_df[energy_col].min(), test_df[energy_col].max())
    overlap_start = max(train_range[0], test_range[0])
    overlap_end = min(train_range[1], test_range[1])
    
    if overlap_end > overlap_start:
        overlap_fraction = (overlap_end - overlap_start) / (max(train_range[1], test_range[1]) - min(train_range[0], test_range[0]))
        print(f"Energy range overlap: {overlap_fraction:.1%}")
    else:
        print("No energy range overlap between train and test!")
    
    return stats_data


def sequential_energy_split(df,
                            energy_col='E',
                            train_size=0.3,
                            val_size=0.3,
                            test_size=0.4,
                            overlap_fraction=0.1,
                            random_state=42):
    """
    Split data sequentially by energy ranges with controlled overlap between splits.
    
    Args:
        df: DataFrame with all data
        energy_col: Name of energy column
        train_size, val_size, test_size: Size proportions (should sum to 1.0)
        overlap_fraction: Fraction of overlap between adjacent splits (e.g., 0.1 = 10%)
        random_state: For reproducibility within each energy range
    
    Returns:
        train_df, val_df, test_df: DataFrames with overlapping energy ranges
    """
    
    # Sort by energy to ensure proper sequential splitting
    df_sorted = df.sort_values(energy_col).reset_index(drop=True)
    
    # Calculate target sample sizes
    n_total = len(df_sorted)
    n_train = int(train_size * n_total)
    n_val = int(val_size * n_total)
    n_test = int(test_size * n_total)
    
    print(f"Sequential energy split with {overlap_fraction:.1%} overlap:")
    print(f"Total samples: {n_total}")
    print(f"Target - Train: {n_train} ({train_size:.1%}), Val: {n_val} ({val_size:.1%}), Test: {n_test} ({test_size:.1%})")
    
    # Calculate overlap sizes
    overlap_train_val = int(overlap_fraction * min(n_train, n_val))
    overlap_val_test = int(overlap_fraction * min(n_val, n_test))
    
    print(f"Overlap sizes - Train-Val: {overlap_train_val}, Val-Test: {overlap_val_test}")
    
    # Calculate the range boundaries with overlap
    # We need to expand the middle ranges to accommodate overlap
    
    # Train range: start from beginning
    train_start = 0
    train_end = n_train + overlap_train_val
    
    # Val range: overlaps with train and test
    val_start = n_train - overlap_train_val
    val_end = n_train + n_val + overlap_val_test
    
    # Test range: overlaps with val, goes to end
    test_start = n_train + n_val - overlap_val_test
    test_end = n_total
    
    # Ensure we don't exceed bounds
    train_end = min(train_end, n_total)
    val_start = max(val_start, 0)
    val_end = min(val_end, n_total)
    test_start = max(test_start, 0)
    
    print(f"Index ranges:")
    print(f"Train: [{train_start}:{train_end}] ({train_end - train_start} samples)")
    print(f"Val:   [{val_start}:{val_end}] ({val_end - val_start} samples)")
    print(f"Test:  [{test_start}:{test_end}] ({test_end - test_start} samples)")
    
    # Extract the ranges
    train_candidates = df_sorted.iloc[train_start:train_end].copy()
    val_candidates = df_sorted.iloc[val_start:val_end].copy()
    test_candidates = df_sorted.iloc[test_start:test_end].copy()
    
    # Sample to get exact target sizes
    np.random.seed(random_state)
    
    # Sample exact sizes from each candidate pool
    train_df = train_candidates.sample(n=min(n_train, len(train_candidates)), 
                                      random_state=random_state).reset_index(drop=True)
    val_df = val_candidates.sample(n=min(n_val, len(val_candidates)), 
                                  random_state=random_state + 1).reset_index(drop=True)
    test_df = test_candidates.sample(n=min(n_test, len(test_candidates)), 
                                    random_state=random_state + 2).reset_index(drop=True)
    
    print(f"Actual split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Calculate energy ranges and statistics
    train_energy_range = (train_df[energy_col].min(), train_df[energy_col].max())
    val_energy_range = (val_df[energy_col].min(), val_df[energy_col].max())
    test_energy_range = (test_df[energy_col].min(), test_df[energy_col].max())
    
    print(f"\nEnergy ranges:")
    print(f"Train: [{train_energy_range[0]:.4f}, {train_energy_range[1]:.4f}]")
    print(f"Val:   [{val_energy_range[0]:.4f}, {val_energy_range[1]:.4f}]")
    print(f"Test:  [{test_energy_range[0]:.4f}, {test_energy_range[1]:.4f}]")
    
    print(f"\nEnergy statistics:")
    print(f"Train - Mean: {train_df[energy_col].mean():.4f}, Median: {train_df[energy_col].median():.4f}")
    print(f"Val   - Mean: {val_df[energy_col].mean():.4f}, Median: {val_df[energy_col].median():.4f}")
    print(f"Test  - Mean: {test_df[energy_col].mean():.4f}, Median: {test_df[energy_col].median():.4f}")
    
    # Check for actual overlap in energy values
    train_energy_range = (train_df[energy_col].min(), train_df[energy_col].max())
    val_energy_range = (val_df[energy_col].min(), val_df[energy_col].max())
    test_energy_range = (test_df[energy_col].min(), test_df[energy_col].max())
    
    train_val_overlap_range = (max(train_energy_range[0], val_energy_range[0]), 
                              min(train_energy_range[1], val_energy_range[1]))
    val_test_overlap_range = (max(val_energy_range[0], test_energy_range[0]), 
                             min(val_energy_range[1], test_energy_range[1]))
    
    train_val_overlap_size = max(0, train_val_overlap_range[1] - train_val_overlap_range[0])
    val_test_overlap_size = max(0, val_test_overlap_range[1] - val_test_overlap_range[0])
    
    print(f"\nActual energy overlaps:")
    if train_val_overlap_size > 0:
        print(f"Train-Val overlap: [{train_val_overlap_range[0]:.4f}, {train_val_overlap_range[1]:.4f}] (range: {train_val_overlap_size:.4f})")
    else:
        print(f"Train-Val overlap: None")
        
    if val_test_overlap_size > 0:
        print(f"Val-Test overlap: [{val_test_overlap_range[0]:.4f}, {val_test_overlap_range[1]:.4f}] (range: {val_test_overlap_size:.4f})")
    else:
        print(f"Val-Test overlap: None")
    
    # Calculate energy ratios
    val_train_ratio = val_df[energy_col].mean() / train_df[energy_col].mean()
    test_train_ratio = test_df[energy_col].mean() / train_df[energy_col].mean()
    test_val_ratio = test_df[energy_col].mean() / val_df[energy_col].mean()
    
    print(f"\nEnergy ratios:")
    print(f"Val/Train ratio: {val_train_ratio:.3f}")
    print(f"Test/Train ratio: {test_train_ratio:.3f}")
    print(f"Test/Val ratio: {test_val_ratio:.3f}")
    
    return train_df, val_df, test_df


def load_data_with_sequential_energy_split(path,
                                           feature_cols,
                                           target_cols,
                                           energy_col='E',
                                           train_size=0.3,
                                           val_size=0.3,
                                           test_size=0.4,
                                           overlap_fraction=0.1,
                                           random_state=42,
                                           output_dir=None,
                                           plot_energy_distributions=True):
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

    # Sequential energy split using original energy values
    print(f"\nUsing sequential energy splitting...")
    train_df, val_df, test_df = sequential_energy_split(
        df, original_energy_col, 
        train_size=train_size, val_size=val_size, test_size=test_size,
        overlap_fraction=overlap_fraction, random_state=random_state
    )

    print(f"\nFinal split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

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