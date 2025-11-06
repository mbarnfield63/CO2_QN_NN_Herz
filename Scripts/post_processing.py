import math
import numpy as np
import os
import pandas as pd
import re
import sys
import time


# Variables
raw_data = '../raw_data/CO2_all_states/'
preds_data = 'Data/qn_predictions.csv'
unc_cutoff = 0.4
energy_cutoff = 15000  # cm-1

iso_formats = {
   626: '12C-16O2',
   627: '16O-12C-17O',
   628: '16O-12C-18O',
   727: '12C-17O2',
   728: '17O-12C-18O',
   828: '12C-18O2',

   636: '13C-16O2',
   637: '16O-13C-17O',
   638: '16O-13C-18O',
   737: '13C-17O2',
   738: '17O-13C-18O',
   838: '13C-18O2',
}

col_specs = {
      0:  (12, "int", None),
      1:  (13, "float", 6),
      2:  (7,  "int", None),
      3:  (7,  "int", None),
      4:  (13, "float", 6),
      5:  (13, "sci", 4),
      6:  (3,  "text", None),
      7:  (3,  "text", None),
      8:  (5,  "int", None),
      9:  (4,  "int", None),
      10: (4,  "int", None),
      11: (4,  "int", None),
      12: (7,  "float", 2),
      13: (7,  "int", None),
      14: (4,  "int", None),
      15: (4,  "int", None),
      16: (4,  "int", None),
      17: (4,  "int", None),
      18: (7,  "int", None),
      19: (4,  "int", None),
      20: (4,  "int", None),
      21: (3,  "text", None),
      22: (13,  "float", None),
}

column_names = ['ID', 'E', 'gtot', 'J', 'unc', '??', 'tot_sym', 'e_f',
               'hzb_v1', 'hzb_v2', 'hzb_l2', 'hzb_v3',
               'Trove_coeff', 'AFGL_m1', 'AFGL_m2', 'AFGL_l2', 'AFGL_m3', 'AFGL_r',
               'Trove_v1', 'Trove_v2', 'Trove_v3', 'Source', 'E_Ca']


def load_predictions(file_path=preds_data, unc_cutoff=unc_cutoff, energy_cutoff=energy_cutoff):
   """Load predictions from a CSV file."""
   if not os.path.exists(file_path):
      raise FileNotFoundError(f"The file {file_path} does not exist.")

   df = pd.read_csv(file_path)

   # Length of dataframe
   print(f"Loaded {len(df)} predictions from {file_path}")

   # Apply filters
   df = df[df['energy'] <= energy_cutoff] # Energy
   df = df[df['uncertainty'] <= unc_cutoff] # Uncertainty

   # Length after filtering
   print(f"{len(df)} predictions remain after applying energy and uncertainty cutoffs.")

   return df


def unique_isotopologues(df):
   """Return a list of unique isotopologues in the dataframe."""
   return df['iso'].unique().tolist()


def select_iso(df, iso_number):
   """Select rows corresponding to a specific isotope."""
   iso_preds_df = df[df['iso'] == iso_number].copy()
   return iso_preds_df


def open_iso_file(iso_number, data_dir=raw_data):
   """Open the file for a specific iso."""

   # Isotope file path
   iso_string = iso_formats.get(iso_number)
   if iso_string is None:
      raise ValueError(f"Unknown isotope number: {iso_number}")
   
   file_path = os.path.join(data_dir, f'{iso_string}__Dozen.states.cut')
   
   if not os.path.exists(file_path):
      raise FileNotFoundError(f"The file for {iso_number} does not exist.")
   
   # Open the file into a dataframe
   iso_raw_df = pd.read_csv(file_path, header=None, skiprows=1, sep=r'\s+', names=column_names)
   return iso_raw_df


def replace_with_preds(iso_raw_df, iso_preds_df):
   """
   Match rows based on energy and J quantum number.
   Replace matched rows in iso_raw_df with values from iso_preds_df.
   """
   
   matched_rows = []
   for _, pred_row in iso_preds_df.iterrows():
      matches = iso_raw_df[(iso_raw_df['E'] == pred_row['energy']) & (iso_raw_df['J'] == pred_row['J'])]
      if not matches.empty:
         matched_rows.append(matches)

   # Check all rows are found
   if len(matched_rows) != len(iso_preds_df):
      raise ValueError("Not all predicted rows were matched in the raw data.")
   
   # Replace rows with predictions
   for match, (_, pred_row) in zip(matched_rows, iso_preds_df.iterrows()):
      idx = match.index[0]
      iso_raw_df.at[idx, 'hzb_v1'] = pred_row['hzb_v1']
      iso_raw_df.at[idx, 'hzb_v2'] = pred_row['hzb_v2']
      iso_raw_df.at[idx, 'hzb_l2'] = pred_row['hzb_l2']
      iso_raw_df.at[idx, 'hzb_v3'] = pred_row['hzb_v3']
      iso_raw_df.at[idx, 'AFGL_m1'] = pred_row['AFGL_m1']
      iso_raw_df.at[idx, 'AFGL_m2'] = pred_row['AFGL_m2']
      iso_raw_df.at[idx, 'AFGL_l2'] = pred_row['AFGL_m2'] # AFGL_l2 is same as AFGL_m2
      iso_raw_df.at[idx, 'AFGL_m3'] = pred_row['AFGL_m3']
      iso_raw_df.at[idx, 'AFGL_r'] = pred_row['AFGL_r']

   # Replace unmatched rows with -1 for prediction columns when no match is found for Ca values
   pred_cols = ['AFGL_m1', 'AFGL_m2', 'AFGL_l2', 'AFGL_m3', 'AFGL_r']
   ca_mask = iso_raw_df['Source'].str.contains('Ca', na=False)
   unmatched_mask = ~iso_raw_df.index.isin([row.index[0] for row in matched_rows]) & ca_mask
   iso_raw_df.loc[unmatched_mask, pred_cols] = -1

   return iso_raw_df


def format_value(val, spec):
   width, ftype, dec_override = spec

   if pd.isna(val) or (isinstance(val, float) and math.isnan(val)):
      s = "NaN"
   else:
      if ftype == "int":
         s = f"{int(val)}"
      elif ftype == "float":
         decimals = dec_override if dec_override is not None else 6
         s = f"{float(val):.{decimals}f}"
      elif ftype == "sci":
         decimals = dec_override if dec_override is not None else 4
         s = f"{float(val):.{decimals}E}"
      else:  # text
         s = str(val)

   # always right align
   s = s.rjust(width)

   # truncate if too long
   return s[:width]


def save_updated_file(iso_raw_df, iso_number, col_specs):
   iso_string = iso_formats.get(iso_number, f"ISO{iso_number}")
   output_path = f"Data/Completed_States/{iso_string}__Dozen_ML.states.cut"
   os.makedirs(os.path.dirname(output_path), exist_ok=True)

   with open(output_path, "w", newline="\n") as f:
      for _, row in iso_raw_df.iterrows():
         line = "".join(format_value(row.iloc[i], col_specs[i]) for i in col_specs)
         f.write(line.rstrip() + "\n")

   print(f"Updated file saved to {output_path}")


if __name__ == "__main__":
   start = time.time()

   if len(sys.argv) > 1:
      iso = int(sys.argv[1])
      if iso not in iso_formats:
         print(f"Invalid isotope number provided: {iso}. Valid options are: {list(iso_formats.keys())}")
         sys.exit(1)
      print(f"Processing only isotope {iso} as specified in command line argument.")
      isotopologues = [iso]
   else:
      print("No isotope number provided. Processing all isotopologues.")
      preds = load_predictions()
      isotopologues = unique_isotopologues(preds)

   preds = load_predictions()

   for iso in isotopologues:
      print(f"Processing isotope: {iso} ({iso_formats[iso]})")
      iso_preds = select_iso(preds, iso)
      iso_raw_df = open_iso_file(iso)
      updated_iso_df = replace_with_preds(iso_raw_df, iso_preds)
      save_updated_file(updated_iso_df, iso, col_specs)

   print("All isotopologues processed successfully.")

   end = time.time()
   print(f"Total time taken: {end - start:.2f} seconds")