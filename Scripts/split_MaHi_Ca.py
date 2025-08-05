import pandas as pd
import glob
import sys
import time

column_names = ['ID', 'E', 'gtot', 'J', 'unc', '??', 'tot_sym', 'e_f',
               'hzb_v1', 'hzb_v2', 'hzb_l2', 'hzb_v3',
               'Trove_coeff', 'AFGL_m1', 'AFGL_m2', 'AFGL_l2', 'AFGL_m3', 'AFGL_r',
               'Trove_v1', 'Trove_v2', 'Trove_v3', 'Source', 'E_Ca']

def split_raw_file(filename):
    """
    Splits the raw file into two separate files:
    - CO2_[iso]_ma.csv = Marvel data
    - CO2_[iso]_ca.csv = Calculated data
    """
    # Get iso from filename
    iso = filename.split('20250617_CO2_')[1].split('_syurchenko.states')[0]
    print(f"Processing {iso}...")

    # Read the file with the specified column names
    df = pd.read_csv(filename, header=None, skiprows=1, sep=r'\s+', names=column_names)
    
    # Clean up any leading/trailing whitespace in string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    marvel_hitran = df[df['Source'].isin(['Ma', 'MA', 'Hi', 'HI'])]
    calculated = df[df['Source'].isin(['Ca', 'CA', 'Eh', 'EH'])]

    # Check combined number of rows = total number of rows
    if len(marvel_hitran) + len(calculated) != len(df):
        raise ValueError("Row counts do not match after splitting.")
    
        # Save the split dataframes to separate files
    print(f"Saving Ma/Hi & Ca for {iso}...")
    marvel_hitran.to_csv(f'Data/Processed/CO2_{iso}_ma.txt', index=False, sep='\t')
    calculated.to_csv(f'Data/Processed/CO2_{iso}_ca.txt', index=False, sep='\t')

    print(f"CO2 {iso} split successfully.")


if __name__ == "__main__":
    time_start = time.time()
    if len(sys.argv) != 2:
        print("Usage: python split_MaHi_Ca.py <data_directory>")
        sys.exit(1)

    directory = sys.argv[1]
    # Get all files in directory that are .states files
    files = glob.glob(f"{directory}/*_syurchenko.states")
    
    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(executor.map(split_raw_file, files))

    print(f"All files processed in {time.time() - time_start:.2f} seconds.")
