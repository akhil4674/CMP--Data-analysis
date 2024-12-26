import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture

# Load the numerical data DataFrame from the CSV file
numerical_data = pd.read_csv("/Users/akhilkumar/Downloads/UMAP/numerical_data.csv")

# Load the UMAP results CSV file
umap_df = pd.read_csv("/Users/akhilkumar/Downloads/UMAP/umap_results.csv")

# Check if the 'Cluster' column exists
if 'Cluster' not in umap_df.columns:
    print("The 'Cluster' column is not present in the UMAP results file. Performing GMM clustering now.")
    
    # Perform GMM clustering
    gmm = GaussianMixture(n_components=5, random_state=42)  # Adjust n_components as needed
    gmm_labels = gmm.fit_predict(umap_df[['UMAP 1', 'UMAP 2', 'UMAP 3']])  # Using UMAP data
    
    # Add GMM labels to the UMAP DataFrame
    umap_df['Cluster'] = gmm_labels
    
    # Save the updated UMAP DataFrame with cluster labels to a CSV file
    umap_df.to_csv("/Users/akhilkumar/Downloads/UMAP/umap_results_with_clusters.csv", index=False)
else:
    # Extract GMM labels
    gmm_labels = umap_df['Cluster'].values

# Convert the data to NumPy arrays for processing
numerical_data_np = numerical_data.to_numpy()

# Add the 'Cluster' column from GMM labels to the numerical data
numerical_data_with_labels_np = np.hstack([numerical_data_np, gmm_labels.reshape(-1, 1)])

# Convert the NumPy array back to pandas for plotting
numerical_data_with_labels = pd.DataFrame(numerical_data_with_labels_np, columns=numerical_data.columns.tolist() + ['Cluster'])

# Create an interactive plot with dropdown to select features
fig = go.Figure()

# Add traces for each feature
for feature in numerical_data.columns:
    fig.add_trace(go.Box(
        x=numerical_data_with_labels['Cluster'],
        y=numerical_data_with_labels[feature],
        name=feature,
        visible=False
    ))

# Make the first feature visible
fig.data[0].visible = True

# Create dropdown menu for box plots
dropdown_buttons = [
    {'label': feature, 'method': 'update', 'args': [{'visible': [feature == trace.name for trace in fig.data]}, {'title': f"Distribution of {feature} Across Clusters"}]}
    for feature in numerical_data.columns
]

# Create a parallel coordinates plot for the features
fig_parallel = go.Figure(data=go.Parcoords(
    line=dict(color=numerical_data_with_labels['Cluster']),
    dimensions=[{'label': col, 'values': numerical_data_with_labels[col]} for col in numerical_data.columns]
))

# Add the parallel coordinates plot as a trace
fig.add_trace(go.Scatter(
    visible=False,
    showlegend=False,
    x=[None],  # Dummy data to initialize the trace
    y=[None]
))

# Update layout with dropdown menu and button to switch to parallel coordinates plot
fig.update_layout(
    updatemenus=[
        {
            'buttons': dropdown_buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.17,
            'xanchor': 'left',
            'y': 1.15,
            'yanchor': 'top'
        },
        {
            'buttons': [
                {'label': 'Box Plot', 'method': 'update', 'args': [{'visible': [True] * len(numerical_data.columns) + [False]}, {'title': 'Distribution of Features Across Clusters'}]},
                {'label': 'Parallel Coordinates', 'method': 'update', 'args': [{'visible': [False] * len(numerical_data.columns) + [True]}, {'title': 'Parallel Coordinates Plot by Cluster'}]}
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.37,
            'xanchor': 'left',
            'y': 1.15,
            'yanchor': 'top'
        }
    ],
    title="Distribution of Features Across Clusters",
    xaxis_title="Cluster",
    yaxis_title="Value"
)

# Show the plot
fig.show()