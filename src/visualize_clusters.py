import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), "../data/mall_customers.csv")
df = pd.read_csv(data_path)

# Load model and scaler
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "../models")

kmeans = joblib.load(os.path.join(models_dir, "kmeans_model.pkl"))
scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

# Scale features and predict clusters
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X_scaled = scaler.transform(df[features])
df['Cluster'] = kmeans.predict(X_scaled)

# Visualization
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set2',
    data=df,
    s=100
)
plt.title("Customer Segmentation (K-Means Clusters)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title='Cluster')
plt.show()
