import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from model_utils import get_predictions


def analyze_energy_performance(model, test_loader, test_df, TARGET_COLS, energy_col='E_original', device='cpu'):
    """Analyze model performance across energy ranges."""
    y_true, y_pred, _, _ = get_predictions(model, test_loader, device)
    
    # Add predictions to test dataframe for analysis
    test_analysis = test_df.copy()
    for i, col in enumerate(TARGET_COLS):
        test_analysis[f'{col}_pred'] = y_pred[:, i]
        test_analysis[f'{col}_correct'] = (y_true[:, i] == y_pred[:, i])
    
    # Analyze performance by energy quartiles
    energy_quartiles = pd.qcut(test_analysis[energy_col], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    test_analysis['energy_quartile'] = energy_quartiles
    
    print("\nPerformance by Energy Quartile:")
    for col in TARGET_COLS:
        accuracy_by_quartile = test_analysis.groupby('energy_quartile')[f'{col}_correct'].mean()
        print(f"{col}:")
        for quartile, acc in accuracy_by_quartile.items():
            print(f"  {quartile}: {acc:.4f}")
    
    return test_analysis


def analyze_isotopologue_predictions(y_true, y_pred, confidences, entropies, df, target_cols):
    """
    Extract predictions for individual isotopologues.
    """    
    isotopologue_mapping = {
        (15.994915, 12.0, 15.994915): "626",
        (15.994915, 12.0, 16.999132): "627",  
        (15.994915, 12.0, 17.999161): "628",
        (16.999132, 12.0, 16.999132): "727",
        (16.999132, 12.0, 17.999161): "728",
        (17.999161, 12.0, 17.999161): "828",        

        (15.994915, 13.003355, 15.994915): "636",
        (15.994915, 13.003355, 16.999132): "637",
        (15.994915, 13.003355, 17.999161): "638",
        (16.999132, 13.003355, 16.999132): "737",
        (16.999132, 13.003355, 17.999161): "738",
        (17.999161, 13.003355, 17.999161): "838",
    }
    
    # Determine which isotopologue each sample represents
    isotopologue_labels = []
    
    for idx in range(len(df)):
        # Find the carbon isotope
        if df.loc[df.index[idx], "mass_c_12.0"] > 0:
            c_mass = 12.0
        else:
            c_mass = 13.003355
            
        # Find the oxygen isotopes
        o1_mass = None
        o2_mass = None
        
        for o_mass in [15.994915, 16.999132, 17.999161]:
            if df.loc[df.index[idx], f"mass_o_1_{o_mass}"] > 0:
                o1_mass = o_mass
            if df.loc[df.index[idx], f"mass_o_2_{o_mass}"] > 0:
                o2_mass = o_mass
                
        # Map to isotopologue name
        isotope_key = (o1_mass, c_mass, o2_mass)
        isotopologue_name = isotopologue_mapping.get(isotope_key, "unknown")
        isotopologue_labels.append(isotopologue_name)
    
    # Group predictions by isotopologue
    results = {}
    isotopologue_labels = np.array(isotopologue_labels)
    
    for isotopologue in np.unique(isotopologue_labels):
        if isotopologue == "unknown":
            continue
            
        mask = isotopologue_labels == isotopologue
        
        # Calculate accuracy for each target column
        accuracies = {}
        for i, label in enumerate(target_cols):
            acc = accuracy_score(y_true[mask, i], y_pred[mask, i])
            accuracies[label] = acc
        
        results[isotopologue] = {
            'y_true': y_true[mask],
            'y_pred': y_pred[mask], 
            'confidences': confidences[mask],
            'entropies': entropies[mask],
            'accuracies': accuracies,
            'count': np.sum(mask)
        }
    
    return results


def save_accuracy_report(results, target_cols, output_dir):
    """
    Save isotopologue accuracy report to CSV.
    
    Parameters:
    -----------
    results : dict
        Output from analyze_isotopologue_predictions
    target_cols : list
        List of target column names
    output_dir : str
        Directory to save the report
    """
    # Ensure the CSVs directory exists
    os.makedirs(os.path.join(output_dir, "CSVs"), exist_ok=True)
    
    # Create a dataframe with isotopologues as rows and target columns as columns
    accuracy_data = {}
    
    for isotopologue, data in results.items():
        accuracy_data[isotopologue] = data['accuracies']
    
    df_accuracies = pd.DataFrame(accuracy_data).T  # Transpose so isotopologues are rows
    df_accuracies.index.name = 'Isotopologue'
    
    # Add sample count column
    df_accuracies['Sample_Count'] = [results[iso]['count'] for iso in df_accuracies.index]
    
    # Save to CSV
    df_accuracies.to_csv(os.path.join(output_dir, "CSVs/isotopologue_accuracy.csv"))
    
    return df_accuracies