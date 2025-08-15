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
    plt.savefig(os.path.join(output_dir, "Plots/Training/loss_plot.png"))
    plt.close()


def plot_enhanced_training_analysis(train_losses, val_losses, train_confidences, val_confidences,
                                    TARGET_COLS, OUTPUT_DIR, confidence_df):
    """
    Plot enhanced training analysis including loss curves, confidence distributions, and confidence vs accuracy.
    """
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

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title("Confidence vs Accuracy")
    ax4.set_xlabel("Confidence Score")
    ax4.set_ylabel("Accuracy")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Plots/Training/enhanced_training_analysis.png"), dpi=300, bbox_inches='tight')
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
        plt.savefig(os.path.join(output_dir, "Plots/Training/detailed_energy_distribution_analysis.png"), 
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


def plot_iso_by_energy_split(train_df, val_df, test_df, iso_col='iso', energy_col='E_original', n_col=3, output_dir=None):
    """
    Plot energy distributions based on the original energy values for each isotopologue.
    """
    # Get unique isotopologues from all splits
    all_isos = sorted(set(train_df[iso_col].unique()) | 
                     set(val_df[iso_col].unique()) | 
                     set(test_df[iso_col].unique()))
    
    # Calculate grid dimensions
    n_isos = len(all_isos)
    n_rows = (n_isos + n_col - 1) // n_col
    
    # Determine global energy range for consistent axis scaling
    all_energies = []
    for df in [train_df, val_df, test_df]:
        all_energies.extend(df[energy_col].values)
    
    energy_min = min(all_energies)
    energy_max = max(all_energies)
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_col, sharex=True, sharey=True, figsize=(5*n_col, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_col == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot colors for each split
    colors = ['blue', 'orange', 'green']
    labels = ['Train', 'Val', 'Test']
    
    for idx, iso in enumerate(all_isos):
        row = idx // n_col
        col = idx % n_col
        ax = axes[row, col]
        
        # Plot histogram for each split
        for data_df, label, color in zip([train_df, val_df, test_df], labels, colors):
            iso_data = data_df[data_df[iso_col] == iso]
            if len(iso_data) > 0:
                ax.hist(iso_data[energy_col], bins=30, alpha=0.3, density=True, 
                       label=f'{label} (n={len(iso_data)})', color=color)
        
        # Set consistent axis limits
        ax.set_xlim(energy_min, energy_max)
        ax.set_xlabel('Energy (cm^-1)')
        ax.set_ylabel('Density')
        ax.set_title(f'Isotopologue: {iso}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_isos, n_rows * n_col):
        row = idx // n_col
        col = idx % n_col
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(os.path.join(output_dir, "Plots"), exist_ok=True)
        plt.savefig(os.path.join(output_dir, "Plots/Isotopologue/isotopologue_energy_distributions.png"), 
                   dpi=300, bbox_inches='tight')
    plt.close()


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
    plt.savefig(os.path.join(output_dir, "Plots/Confidence/confidence_by_energy.png"), dpi=300, bbox_inches='tight')
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
    plt.savefig(os.path.join(output_dir, "Plots/Confidence/confidence_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_isotopologue_accuracies(results, target_cols, output_dir, figsize=(12, 8)):
    """
    Create bar plots showing accuracy for each isotopologue across different target columns.
    
    Parameters:
    -----------
    results : dict
        Output from analyze_isotopologue_predictions
    target_cols : list
        List of target column names
    output_dir : str
        Directory to save the plots
    figsize : tuple
        Figure size for each subplot
    """
    # Ensure plots directory exists
    os.makedirs(os.path.join(output_dir, "Plots"), exist_ok=True)
    
    # Extract isotopologue names and sort them
    isotopologues = sorted(results.keys())
    
    # Create a separate plot for each target column
    for i, target_col in enumerate(target_cols):
        plt.figure(figsize=figsize)
        
        # Extract accuracies for this target column
        accuracies = [results[iso]['accuracies'][target_col] for iso in isotopologues]
        sample_counts = [results[iso]['count'] for iso in isotopologues]
        
        # Create bar plot
        bars = plt.bar(isotopologues, accuracies, alpha=0.7, color=plt.cm.viridis(i/len(target_cols)))
        
        # Add sample count annotations on top of bars
        for j, (bar, count) in enumerate(zip(bars, sample_counts)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'n={count}', ha='center', va='bottom', fontsize=9)
        
        plt.title(f'Accuracy by Isotopologue - {target_col}', fontsize=14, fontweight='bold')
        plt.xlabel('Isotopologue (OCO notation)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', alpha=0.3)
        
        # Add horizontal line at mean accuracy
        mean_acc = np.mean(accuracies)
        plt.axhline(y=mean_acc, color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean: {mean_acc:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Plots/Isotopologue/isotopologue_accuracy_{target_col.replace('/', '_')}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def plot_isotopologue_comparison(results, target_cols, output_dir, figsize=(15, 10)):
    """
    Create a comprehensive comparison plot showing all target columns for all isotopologues.
    
    Parameters:
    -----------
    results : dict
        Output from analyze_isotopologue_predictions
    target_cols : list
        List of target column names
    output_dir : str
        Directory to save the plots
    figsize : tuple
        Figure size
    """
    # Ensure plots directory exists
    os.makedirs(os.path.join(output_dir, "Plots"), exist_ok=True)
    
    # Prepare data for plotting
    isotopologues = sorted(results.keys())
    
    # Create dataframe for easier plotting with seaborn
    plot_data = []
    for iso in isotopologues:
        for target in target_cols:
            plot_data.append({
                'Isotopologue': iso,
                'Target': target,
                'Accuracy': results[iso]['accuracies'][target],
                'Sample_Count': results[iso]['count']
            })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create the comparison plot
    plt.figure(figsize=figsize)
    
    # Create grouped bar plot
    x = np.arange(len(isotopologues))
    width = 0.8 / len(target_cols)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(target_cols)))
    
    for i, target in enumerate(target_cols):
        target_data = df_plot[df_plot['Target'] == target]
        accuracies = [target_data[target_data['Isotopologue'] == iso]['Accuracy'].iloc[0] 
                     for iso in isotopologues]
        
        plt.bar(x + i * width - width * (len(target_cols)-1)/2, accuracies, 
               width, label=target, color=colors[i], alpha=0.8)
    
    plt.xlabel('Isotopologue (OCO notation)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy Comparison Across All Isotopologues and Target Columns', 
              fontsize=14, fontweight='bold')
    plt.xticks(x, isotopologues)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Plots/Isotopologue/all_isotopologues_accuracy_comparison.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(df, output_dir):
    # Ensure the Plots directory exists
    os.makedirs(os.path.join(output_dir, "Plots"), exist_ok=True)
    
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12), sharex=False)
    axes = axes.flatten()

    # Plot the feature importance for each quantum number
    for i, col in enumerate(df.columns):
        # Sort the features by importance for the current quantum number
        sorted_df = df[[col]].sort_values(by=col, ascending=True)
        
        # Plotting on the current axis
        ax = axes[i]
        sorted_df.plot(kind='barh', ax=ax, legend=False)
        ax.set_title(f"Feature Importance for {col}")
        ax.set_xlabel("Importance Score (Drop in Accuracy)")
        ax.set_ylabel("Feature")
        
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()

    # Save the combined plot to the output directory
    plt.savefig(os.path.join(output_dir, "Plots/Features/feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()