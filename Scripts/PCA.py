import pandas as pd
from analysis import PCA_analysis

# Load training data
data_path = "Data/CO2_all_ma_1.txt"
data = pd.read_csv(data_path)

features = ['hzb_v1', 'hzb_v2', 'hzb_v3', 'hzb_l2',
            'AFGL_m1', 'AFGL_m2', 'AFGL_m3', 'AFGL_r',
            'Trove_coeff', 'Trove_v1', 'Trove_v2', 'Trove_v3']

# Run PCA
pca_results = PCA_analysis(data, features)

# Optionally, print or save results
print(pca_results)