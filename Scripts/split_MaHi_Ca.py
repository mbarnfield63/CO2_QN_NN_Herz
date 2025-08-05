import pandas as pd
import glob
import sys


def split_raw_file(filename):
    """
    Splits the raw file into two separate files:
    - CO2_[iso]_ma.csv = Marvel data
    - CO2_[iso]_ca.csv = Calculated data

    Returns:
    None
    """
    # Import raw file
    print(f"Importing raw file: {filename}...")
    df = pd.read_csv(filename, header=None, skiprows=1, sep=r'\s+')

    marvel_hitran = df[df[21].isin(['Ma', 'MA', 'Hi', 'HI'])]
    calculated = df[df[21].isin(['Ca', 'CA', 'Eh', 'EH'])]

    # Check combined number of rows = total number of rows
    if len(marvel_hitran) + len(calculated) != len(df):
        raise ValueError("Row counts do not match after splitting.")
    
    # Get iso from filename
    iso = filename.split('20250617_CO2_')[1].split('_syurchenko.states')[0]

    # Save the split dataframes to separate files
    print("Saving Ma/Hi file...")
    marvel_hitran.to_csv(f'Data/Processed/CO2_{iso}_ma.txt',
                         index=False, header=False, sep='\t')
    print("Saving Ca file...")
    calculated.to_csv(f'Data/Processed/CO2_{iso}_ca.txt', index=False,
                      header=False, sep='\t')

    print(f"CO2 {iso} split successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python raw_split.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    # Get all files in directory that are .states files
    files = glob.glob(f"{directory}/*_syurchenko.states")
    
    for filename in files:
        print(f"Processing file: {filename}")
        split_raw_file(filename)
