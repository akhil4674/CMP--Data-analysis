import pandas as pd
from sklearn.mixture import GaussianMixture
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_table

# Load the UMAP DataFrame from the CSV file
umap_df = pd.read_csv("/Users/akhilkumar/Downloads/UMAP/umap_results.csv")

# Load the numerical data DataFrame from the CSV file
numerical_data = pd.read_csv("/Users/akhilkumar/Downloads/UMAP/numerical_data.csv")

# --- GMM Clustering ---
# Fit Gaussian Mixture Model (GMM) for clustering
gmm = GaussianMixture(n_components=5, random_state=42)  # Adjust n_components as needed
gmm_labels = gmm.fit_predict(umap_df[['UMAP 1', 'UMAP 2', 'UMAP 3']])  # Using UMAP data

# Add GMM labels to the UMAP DataFrame
umap_df['Cluster'] = gmm_labels

# --- Visualizing with Plotly ---
# Sample a subset of the UMAP data for plotting to reduce computational load
sampled_umap_df = umap_df.sample(n=1000, random_state=42)  # Adjust the sample size as needed

# Create an interactive 3D scatter plot of the UMAP data colored by clusters
fig = px.scatter_3d(sampled_umap_df, x='UMAP 1', y='UMAP 2', z='UMAP 3', color='Cluster',
                    hover_data={'Cluster': True, 'UMAP 1': True, 'UMAP 2': True, 'UMAP 3': True},
                    title="Interactive 3D UMAP Projection with GMM Clusters")

# --- Cluster Centroid Calculations ---
# Step 1: Calculate the means (centroids) of the clusters in the UMAP space
cluster_centroids_umap = umap_df.groupby('Cluster')[['UMAP 1', 'UMAP 2', 'UMAP 3']].mean()

# Step 2: Calculate the means of the clusters in the original feature space
numerical_data_with_labels = numerical_data.copy()
numerical_data_with_labels['Cluster'] = gmm_labels

# Compute the mean of each cluster for the original features
cluster_centroids_features = numerical_data_with_labels.groupby('Cluster').mean()

# Create a Dash application
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(figure=fig),
    html.H2("UMAP Space Cluster Centroids"),
    dash_table.DataTable(
        data=cluster_centroids_umap.reset_index().to_dict('records'),
        columns=[{"name": i, "id": i} for i in cluster_centroids_umap.reset_index().columns],
        style_table={'height': '300px', 'overflowY': 'auto'},
    ),
    html.H2("Original Feature Space Cluster Centroids"),
    dash_table.DataTable(
        data=cluster_centroids_features.reset_index().to_dict('records'),
        columns=[{"name": i, "id": i} for i in cluster_centroids_features.reset_index().columns],
        style_table={'height': '300px', 'overflowY': 'auto'},
    ),
])

if __name__ == '__main__':
    app.run_server(debug=True)