import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('Data/hzb_predictions_with_uncertainty.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Create scatter plot without hue
sns.scatterplot(data=df, x='energy', y='uncertainty', ax=ax, alpha=0.6)

# Set y-axis limits with top at 1.0
ax.set_ylim(0, 1.0)

# Add horizontal lines at each 0.1 interval
y_intervals = np.arange(0, 1.1, 0.1)
for y_val in y_intervals:
    ax.axhline(y=y_val, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

# Calculate and display percentages for each 0.1 region
total_points = len(df)
for i in range(len(y_intervals) - 1):
    lower_bound = y_intervals[i]
    upper_bound = y_intervals[i + 1]
    
    # Count points in this region
    points_in_region = ((df['uncertainty'] >= lower_bound) & 
                       (df['uncertainty'] < upper_bound)).sum()
    
    # Calculate percentage
    percentage = (points_in_region / total_points) * 100
    
    # Add text annotation on the right side of the plot
    y_position = (lower_bound + upper_bound) / 2
    ax.text(ax.get_xlim()[1] * 1.02, y_position, f'{percentage:.3f}%', 
            verticalalignment='center', fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Set labels and title
ax.set_title('Uncertainty vs Energy with Regional Percentages')
ax.set_xlabel('Energy (cm-1)')
ax.set_ylabel('Uncertainty')

# Adjust layout to make room for percentage labels
plt.tight_layout()
plt.subplots_adjust(right=0.85)

plt.savefig('Data/Outputs/Plots/uncertainty_vs_energy.png', bbox_inches='tight', dpi=300)