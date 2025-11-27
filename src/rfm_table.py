import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. LOAD DATA ---
file_path = r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\data\cleaned_online_retail.csv'
df_clean = pd.read_csv(file_path)

# CRITICAL FIX: Convert InvoiceDate to datetime
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

# --- 3. RFM CALCULATION ---
# Create Snapshot Date (The day after the last purchase in the dataset)
snapshot_date = df_clean['InvoiceDate'].max() + dt.timedelta(days=1)
print(f"Snapshot Date: {snapshot_date}")

# Group by CustomerID to calculate Recency, Frequency, and Monetary
rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                                   # Frequency
    'TotalAmount': 'sum'                                      # Monetary
}).reset_index()

# Rename columns
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# --- 4. INSPECTION ---
print("\nRFM Table (First 5 Rows):")
print(rfm.head())

print("\nRFM Stats:")
print(rfm.describe())

# Save the RFM table for further use
rfm.to_csv(r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\data\rfm_table.csv', index=False)

# --- 5. VISUALIZE ORIGINAL DISTRIBUTION ---
plt.figure(figsize=(15, 5))

# Plot Recency
plt.subplot(1, 3, 1)
sns.histplot(data=rfm, x='Recency', kde=True, color='skyblue')
plt.title('Recency Distribution')
plt.xlabel('Days Since Last Purchase')

# Plot Frequency
plt.subplot(1, 3, 2)
sns.histplot(data=rfm, x='Frequency', kde=True, color='lightgreen')
plt.title('Frequency Distribution')
plt.xlabel('Number of Orders')

# Plot Monetary
plt.subplot(1, 3, 3)
sns.histplot(data=rfm, x='Monetary', kde=True, color='lightcoral')
plt.title('Monetary Distribution')
plt.xlabel('Total Amount Spent')

plt.tight_layout()

# Save the plot
plt.savefig(r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\figures\rfm_distribution.png', dpi=300, bbox_inches='tight')
print("\nOriginal RFM distribution plot saved!")
plt.close()  # Close the figure to free memory

# --- 6. LOG TRANSFORMATION ---
# Filter out zero or negative Monetary values before log transformation
rfm_clean = rfm[rfm['Monetary'] > 0].copy()

print(f"\nRows before filtering: {len(rfm)}")
print(f"Rows after filtering (Monetary > 0): {len(rfm_clean)}")

# Apply Log Transformation to all RFM metrics
# Note: We add 1 to Recency to avoid log(0) issues
rfm_log = rfm_clean.copy()
rfm_log['Recency'] = np.log(rfm_clean['Recency'] + 1)
rfm_log['Frequency'] = np.log(rfm_clean['Frequency'])
rfm_log['Monetary'] = np.log(rfm_clean['Monetary'])

# Check the new distribution stats
print("\nLog Transformed Stats:")
print(rfm_log.describe())

# Save the log-transformed RFM table for further use
rfm_log.to_csv(r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\data\rfm_log_table.csv', index=False)

# --- 7. VISUALIZE LOG-TRANSFORMED DISTRIBUTION ---
plt.figure(figsize=(15, 5))

# Plot Log Recency
plt.subplot(1, 3, 1)    
sns.histplot(data=rfm_log, x='Recency', kde=True, color='skyblue')
plt.title('Log Recency Distribution')
plt.xlabel('Log(Recency + 1)')

# Plot Log Frequency
plt.subplot(1, 3, 2)
sns.histplot(data=rfm_log, x='Frequency', kde=True, color='lightgreen')
plt.title('Log Frequency Distribution')
plt.xlabel('Log(Frequency)')

# Plot Log Monetary
plt.subplot(1, 3, 3)
sns.histplot(data=rfm_log, x='Monetary', kde=True, color='lightcoral')
plt.title('Log Monetary Distribution')
plt.xlabel('Log(Monetary)')

plt.tight_layout()

# Save the plot
plt.savefig(r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\figures\rfm_log_distribution.png', dpi=300, bbox_inches='tight')
print("\nLog-transformed RFM distribution plot saved!")
plt.close()  # Close the figure to free memory

print("\nRFM Analysis Complete!")