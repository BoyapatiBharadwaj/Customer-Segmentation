import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), "../data/Mall_Customers.csv")
df = pd.read_csv(data_path)

# Select relevant features
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using Elbow method (optional)
# Here we set 5 clusters directly for simplicity
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# Assign cluster labels
df['Cluster'] = kmeans.labels_

# Save model and scaler
# Safe path handling
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "../models")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(kmeans, os.path.join(models_dir, "kmeans_model.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

print(f"✅ K-Means model trained with {k} clusters!")
print(df.groupby('Cluster').size())
