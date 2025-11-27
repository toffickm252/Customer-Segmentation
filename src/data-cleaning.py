import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. LOAD DATA ---
file_path = r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\data\Online Retail.xlsx'
data = pd.read_excel(file_path)

# --- 2. DATA CLEANING ---
# Filter: Remove negative quantities and missing Customer IDs
df_clean = data[(data['Quantity'] > 0) & (data['CustomerID'].notna())].copy()

# Feature Creation: Calculate TotalAmount for each line item
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

print(f"Original shape: {data.shape}")
print(f"Cleaned shape:  {df_clean.shape}")

# save cleaned data for further use
df_clean.to_csv(r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\data\cleaned_online_retail.csv', index=False)

