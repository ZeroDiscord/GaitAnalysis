import os
import pandas as pd

base_path = r"C:\Users\ASUS\Desktop\dataset\Research Oriented\Gaitab\Datasets"
folders = ['1_Healthy', '3_Hemiplegia', '5_PIVD_RA', '8_Osteoarthiritis']

for f in folders:
    f_path = os.path.join(base_path, f)
    if os.path.isdir(f_path):
        files = [x for x in os.listdir(f_path) if x.endswith('.csv')]
        if files:
            first_file = os.path.join(f_path, files[0])
            df = pd.read_csv(first_file)
            print(f"\n--- {f} / {files[0]} ---")
            print("Shape:", df.shape)
            print(df.head())
