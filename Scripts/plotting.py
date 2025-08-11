import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


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
        plt.savefig(os.path.join(output_dir, f"Plots/Confusion/confusion_{label}.png"))
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
        print(f"  Std:        {energy_data.std():>8.4f}")
        print(f"  Min:        {energy_data.min():>8.4f}")
        print(f"  Max:        {energy_data.max():>8.4f}")

    
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


def confidence_by_energy(y_true, y_pred, confidences, entropies, 
                         energy_values, target_cols, output_dir):
    # Sort by energy for smooth lines
    sort_idx = np.argsort(energy_values)
    energy_sorted = energy_values[sort_idx]

    os.makedirs(os.path.join(output_dir, "CSVs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Plots"), exist_ok=True)

    # Prepare DataFrame for CSV output
    records = []
    for i, target in enumerate(target_cols):
        for idx in sort_idx:
            records.append({
                'target': target,
                'energy': energy_values[idx],
                'confidence': confidences[idx, i],
                'entropy': entropies[idx, i],
                'correct': int(y_true[idx, i] == y_pred[idx, i])
            })
    df_analysis = pd.DataFrame(records)
    df_analysis.to_csv(os.path.join(output_dir, "CSVs/confidence_by_energy.csv"), index=False)

    # Plot confidence and accuracy as a function of energy (line plots)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, target in enumerate(target_cols):
        ax = axes[i]
        # Get sorted values for this target
        conf_sorted = confidences[sort_idx, i]
        ent_sorted = entropies[sort_idx, i]
        correct_sorted = (y_true[sort_idx, i] == y_pred[sort_idx, i]).astype(float)

        # Rolling mean for accuracy for smoother curve
        window = max(1, len(energy_sorted) // 100)
        if window > 1:
            acc_smooth = pd.Series(correct_sorted).rolling(window, center=True, min_periods=1).mean().values
            conf_smooth = pd.Series(conf_sorted).rolling(window, center=True, min_periods=1).mean().values
        else:
            acc_smooth = correct_sorted
            conf_smooth = conf_sorted

        ax.plot(energy_sorted, acc_smooth, label='Accuracy', color='blue', alpha=0.7)
        ax.plot(energy_sorted, conf_smooth, label='Confidence', color='red', alpha=0.7)

        ax.set_xlabel('Energy')
        ax.set_ylabel('Score')
        ax.set_title(f'{target}')
        ax.grid(True, alpha=0.3)

        # Set y-axis limits
        ax.set_ylim(0, 1.05)

        # Add legend
        ax.legend(loc='lower left')

    plt.suptitle("Confidence & Accuracy by Energy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Plots/confidence_by_energy.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_distribution(confidences, entropies, target_cols, output_dir):
    """
    Plot distribution of confidence scores for each target
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, target in enumerate(target_cols):
        ax = axes[i]
        
        # Plot histogram of confidence scores
        ax.hist(confidences[:, i], bins=50, alpha=0.7, label='Confidence', density=True)
        ax.axvline(np.mean(confidences[:, i]), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(confidences[:, i]):.3f}')
        
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{target}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Confidence Distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Plots/confidence_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()