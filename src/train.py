import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
rfm_log = pd.read_csv(r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\data\rfm_log_table.csv')

# set CustomerID as index
rfm_log.set_index('CustomerID', inplace=True)

print("Original RFM Log Data:")
print(rfm_log.head())
print(f"Number of customers in log data: {len(rfm_log)}")

# Standardize the data
# Initialize the Scaler
scaler = StandardScaler()

# Fit and Transform
# This calculates the mean/std and converts the data
scaler.fit(rfm_log)
rfm_scaled = scaler.transform(rfm_log)

# Create a clean DataFrame
# We put the column names and index back so it's easy to read
rfm_scaled = pd.DataFrame(rfm_scaled, index=rfm_log.index, columns=rfm_log.columns)

print("\nScaled Data (Mean ≈ 0, Std ≈ 1):")
print(rfm_scaled.head())
print(rfm_scaled.describe().round(2))

# save scaled to joblib file
import joblib
joblib.dump(scaler, r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\models\scaler.joblib')
print("\nScaler saved to 'scaler.joblib'")

# Save the scaled data for further use
rfm_scaled.to_csv(r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\data\rfm_scaled_table.csv')
print("\nScaled data saved to 'rfm_scaled_table.csv'")

# Create a loop that runs K-Means for k = 1 to k = 10
# Initialize an empty list to store the inertia values
inertia = []

# Loop through k = 1 to k = 10
for k in range(1, 11):
    # 1. Initialize KMeans with k clusters
    # random_state=42 ensures we get the same result every time
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    # 2. Fit the model to the scaled data
    kmeans.fit(rfm_scaled)
    
    # 3. Store the inertia (error)
    inertia.append(kmeans.inertia_)

# save inertia values for further analysis
inertia_df = pd.DataFrame({'k': range(1, 11), 'inertia': inertia})
inertia_df.to_csv(r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\data\kmeans_inertia_values.csv', index=False)
print("\nInertia values saved to 'kmeans_inertia_values.csv'")

# Plot the inertia values to visualize the elbow
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker='o', linewidth=2, markersize=8, color='blue')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.xticks(range(1, 11))
plt.grid(True, alpha=0.3)

plt.savefig(r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\figures\kmeans_elbow_method.png', dpi=300, bbox_inches='tight')
print("\nElbow method plot saved!")

plt.close()  # Close the figure to free memory

# --- 5. RUN K-MEANS ---
# Initialize with k=2
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

# Fit the model to the scaled data
kmeans.fit(rfm_scaled)

# Assign the labels (0 or 1) back to our LOG dataframe
rfm_log['Cluster'] = kmeans.labels_

# Check the counts
print("\nCluster Sizes:")
print(rfm_log['Cluster'].value_counts())

# Save the RFM LOG table with cluster labels for further use
rfm_log.to_csv(r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\data\rfm_log_with_clusters.csv')
print("\nRFM log table with cluster labels saved to 'rfm_log_with_clusters.csv'")

# --- 6. CLUSTER INTERPRETATION ---
# 1. Load the original (interpretable) data
rfm_original = pd.read_csv(r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\data\rfm_table.csv')

# 2. Set the index (so it matches the order of our model)
rfm_original.set_index('CustomerID', inplace=True)

print(f"\nNumber of customers in original data: {len(rfm_original)}")

# CRITICAL FIX: Filter rfm_original to only include customers that exist in rfm_log
# This ensures the lengths match
rfm_original_filtered = rfm_original.loc[rfm_log.index]

print(f"Number of customers after filtering: {len(rfm_original_filtered)}")

# 3. Add the cluster labels
# The model's labels match the order of the data we fed it
rfm_original_filtered['Cluster'] = kmeans.labels_

# 4. Analyze the average behavior of each cluster
cluster_summary = rfm_original_filtered.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).round(2)

# Add cluster sizes
cluster_summary['CustomerCount'] = rfm_original_filtered.groupby('Cluster').size()

print("\nCluster Analysis (Averages):")
print(cluster_summary)

# Save the cluster summary for further use
cluster_summary.to_csv(r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\data\cluster_summary.csv')
print("\nCluster summary saved to 'cluster_summary.csv'")

# --- 7. SAVE FINAL RFM TABLE WITH CLUSTERS ---
rfm_original_filtered.to_csv(r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\data\rfm_with_clusters.csv')
print("\nFinal RFM table with clusters saved to 'rfm_with_clusters.csv'")

print(f"\n{'='*60}")
print("CLUSTERING COMPLETE!")
print(f"{'='*60}")
print(f"Total customers clustered: {len(rfm_original_filtered)}")
print(f"Customers excluded (Monetary <= 0): {len(rfm_original) - len(rfm_original_filtered)}")

# save model to joblib file
joblib.dump(kmeans, r'C:\Users\Surface\OneDrive\Documentos\GitHub\Customer-Segmentation\models\kmeans_model.joblib')
print("\nKMeans model saved to 'kmeans_model.joblib'")