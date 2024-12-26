import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
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

# Calculate the means (centroids) of the clusters in the UMAP space
cluster_centroids_umap = umap_df.groupby('Cluster')[['UMAP 1', 'UMAP 2', 'UMAP 3']].mean()

# Calculate the means of the clusters in the original feature space
numerical_data_with_labels['Cluster'] = gmm_labels
cluster_centroids_features = numerical_data_with_labels.groupby('Cluster').mean()

# Truncate values for display
def truncate_values(df, decimals=2):
    return df.applymap(lambda x: f"{x:.{decimals}f}")

# Create the Dash app
app = Dash(__name__, suppress_callback_exceptions=True)

# Layout of the app
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Login page layout
login_layout = html.Div([
    html.Div([
        html.Img(src='https://1000logos.net/wp-content/uploads/2016/10/Bosch-Logo-1024x640.png', style={'height': '140px', 'margin-bottom': '20px'}),
        html.H1("Login to UMAP and GMM Clustering Dashboard", style={'textAlign': 'center', 'font-family': 'Arial', 'color': '#007BFF'}),
        dcc.Input(id='username', type='text', placeholder='Username', style={'margin-bottom': '10px'}),
        dcc.Input(id='password', type='password', placeholder='Password', style={'margin-bottom': '10px'}),
        html.Button('Login', id='login-button', n_clicks=0, style={'margin-bottom': '10px'}),
        html.Div(id='login-output', style={'color': 'red'})
    ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'justify-content': 'center', 'height': '100vh'})
])

# Main dashboard layout
dashboard_layout = html.Div([
    html.Div([
        html.Img(src='https://1000logos.net/wp-content/uploads/2016/10/Bosch-Logo-1024x640.png', style={'height': '140px', 'margin-right': '20px'}),
        html.H1("UMAP and GMM Clustering Dashboard", style={'textAlign': 'center', 'font-family': 'Arial'})
    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'padding': '20px', 'border-bottom': '2px solid #007BFF'}),
    dcc.Markdown("""
        This dashboard provides an interactive way to explore the results of UMAP dimensionality reduction and GMM clustering for the CMP Data,
        Use the tabs below to navigate through different visualizations and insights.
    """, style={'textAlign': 'center', 'font-family': 'Arial', 'padding': '20px'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Box Plot', value='tab-1', style={'font-family': 'Arial'}),
        dcc.Tab(label='Scatter Matrix', value='tab-2', style={'font-family': 'Arial'}),
        dcc.Tab(label='3D Scatter Plot', value='tab-3', style={'font-family': 'Arial'}),
        dcc.Tab(label='Cluster Centroids', value='tab-4', style={'font-family': 'Arial'}),
        dcc.Tab(label='Help', value='tab-5', style={'font-family': 'Arial'})
    ], style={'padding': '20px'}),
    html.Div(id='tabs-content', style={'padding': '20px'}),
    html.Div([
        dcc.Markdown("""
            ### All rights reserved to Robert Bosch GmbH and made by Akhil Kumar ❤️
        """, style={'textAlign': 'center', 'font-family': 'Arial', 'padding': '20px', 'color': '#007BFF'})
    ], style={'border-top': '2px solid #007BFF', 'padding': '20px', 'backgroundColor': '#F8F9FA'})
], style={'backgroundColor': '#F8F9FA', 'padding': '20px'})

# Callback to handle login
@app.callback(
    Output('url', 'pathname'),
    [Input('login-button', 'n_clicks')],
    [State('username', 'value'), State('password', 'value')]
)
def login(n_clicks, username, password):
    if n_clicks > 0:
        if username == 'admin' and password == 'admin':
            return '/dashboard'
        else:
            return '/login'
    return '/login'

# Callback to update the page content based on the URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/dashboard':
        return dashboard_layout
    else:
        return login_layout

# Callback to update the content based on the selected tab
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def render_content(tab):
    if tab == 'tab-1':
        fig = go.Figure()
        for feature in numerical_data.columns:
            fig.add_trace(go.Box(
                x=numerical_data_with_labels['Cluster'],
                y=numerical_data_with_labels[feature],
                name=feature,
                visible=False
            ))
        fig.data[0].visible = True
        dropdown_buttons = [
            {'label': feature, 'method': 'update', 'args': [{'visible': [feature == trace.name for trace in fig.data]}, {'title': f"Distribution of {feature} Across Clusters"}]}
            for feature in numerical_data.columns
        ]
        fig.update_layout(
            updatemenus=[{
                'buttons': dropdown_buttons,
                'direction': 'down',
                'showactive': True,
            }],
            title="Distribution of Features Across Clusters",
            xaxis_title="Cluster",
            yaxis_title="Value",
            font=dict(family="Arial", size=12, color="black"),
            paper_bgcolor='#F8F9FA',
            plot_bgcolor='#F8F9FA'
        )
        return html.Div([
            dcc.Markdown("""
                ### Box Plot
                This plot shows the distribution of each feature across different clusters. Use the dropdown menu to select a feature.
            """, style={'font-family': 'Arial', 'padding': '10px'}),
            dcc.Graph(figure=fig)
        ], style={'border': '1px solid #007BFF', 'padding': '20px', 'border-radius': '10px', 'backgroundColor': '#FFFFFF'})
    elif tab == 'tab-2':
        return html.Div([
            dcc.Markdown("""
                ### Scatter Matrix
                This plot shows the relationships between multiple selected features.
            """, style={'font-family': 'Arial', 'padding': '10px'}),
            dcc.Dropdown(
                id='features-dropdown',
                options=[{'label': col, 'value': col} for col in numerical_data.columns],
                value=numerical_data.columns[:3].tolist(),
                multi=True,
                style={'margin-bottom': '20px'}
            ),
            dcc.Graph(id='scatter-matrix')
        ], style={'border': '1px solid #007BFF', 'padding': '20px', 'border-radius': '10px', 'backgroundColor': '#FFFFFF'})
    elif tab == 'tab-3':
        fig = px.scatter_3d(umap_df, x='UMAP 1', y='UMAP 2', z='UMAP 3', color='Cluster',
                            title="Interactive 3D UMAP Projection with GMM Clusters")
        fig.update_layout(
            font=dict(family="Arial", size=12, color="black"),
            paper_bgcolor='#F8F9FA',
            plot_bgcolor='#F8F9FA',
            height=800  # Increase the height of the plot
        )
        return html.Div([
            dcc.Markdown("""
                ### 3D Scatter Plot
                This plot shows the UMAP-reduced data in a 3D space, colored by clusters.
            """, style={'font-family': 'Arial', 'padding': '10px'}),
            dcc.Graph(figure=fig, style={'height': '800px'}),  # Increase the height of the div
            dcc.Markdown("""
                ### Explanation
                The 3D scatter plot visualizes the UMAP-reduced data in three dimensions. Each point represents a data sample, and the colors indicate the clusters identified by the GMM algorithm.
                This visualization helps to understand the distribution and separation of clusters in the reduced dimensional space.
            """, style={'font-family': 'Arial', 'padding': '10px'})
        ], style={'border': '1px solid #007BFF', 'padding': '20px', 'border-radius': '10px', 'backgroundColor': '#FFFFFF'})
    elif tab == 'tab-4':
        truncated_umap = truncate_values(cluster_centroids_umap)
        truncated_features = truncate_values(cluster_centroids_features)

        table_umap = go.Figure(data=[go.Table(
            header=dict(values=["Cluster", "UMAP 1", "UMAP 2", "UMAP 3"],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[truncated_umap.index, truncated_umap['UMAP 1'], truncated_umap['UMAP 2'], truncated_umap['UMAP 3']],
                       fill_color='lavender',
                       align='left'))
        ])
        table_umap.update_layout(
            title="UMAP Space Cluster Centroids",
            font=dict(family="Arial", size=12, color="black"),
            paper_bgcolor='#F8F9FA',
            plot_bgcolor='#F8F9FA'
        )

        table_features = go.Figure(data=[go.Table(
            header=dict(values=["Cluster"] + list(truncated_features.columns[:5]),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[truncated_features.index] + [truncated_features[col] for col in truncated_features.columns[:5]],
                       fill_color='lavender',
                       align='left'))
        ])
        table_features.update_layout(
            title="Original Feature Space Cluster Centroids",
            font=dict(family="Arial", size=12, color="black"),
            paper_bgcolor='#F8F9FA',
            plot_bgcolor='#F8F9FA'
        )

        heatmap_umap = px.imshow(cluster_centroids_umap.values, labels=dict(x="UMAP Dimensions", y="Cluster", color="Value"),
                                 x=["UMAP 1", "UMAP 2", "UMAP 3"], y=cluster_centroids_umap.index,
                                 title="Heatmap of UMAP Space Cluster Centroids")
        heatmap_features = px.imshow(cluster_centroids_features.values, labels=dict(x="Features", y="Cluster", color="Value"),
                                     x=cluster_centroids_features.columns, y=cluster_centroids_features.index,
                                     title="Heatmap of Original Feature Space Cluster Centroids")

        return html.Div([
            dcc.Markdown("""
                ### Cluster Centroids
                These tables show the centroids of clusters in both the UMAP space and the original feature space.
            """, style={'font-family': 'Arial', 'padding': '10px'}),
            html.Div([
                dcc.Graph(figure=table_umap),
                dcc.Graph(figure=heatmap_umap)
            ], style={'border': '1px solid #007BFF', 'padding': '20px', 'border-radius': '10px', 'backgroundColor': '#FFFFFF', 'margin-bottom': '20px'}),
            dcc.Markdown("""
                ### Original Feature Space Cluster Centroids
                Use the dropdown menu to select features.
            """, style={'font-family': 'Arial', 'padding': '10px'}),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': col, 'value': col} for col in cluster_centroids_features.columns],
                multi=True,
                value=cluster_centroids_features.columns[:5].tolist(),
                style={'margin-bottom': '20px'}
            ),
            html.Div([
                dcc.Graph(id='feature-table'),
                dcc.Graph(id='feature-heatmap')
            ], style={'border': '1px solid #007BFF', 'padding': '20px', 'border-radius': '10px', 'backgroundColor': '#FFFFFF'})
        ])
    elif tab == 'tab-5':
        return html.Div([
            dcc.Markdown("""
                ### Help
                This section provides contact information for further assistance.
            """, style={'font-family': 'Arial', 'padding': '10px', }),
            dcc.Markdown("""
                **Akhil Kumar**

                Process Engineering (RtP1/MSF-EN14)
                
                Robert Bosch GmbH | PO Box 13 42 | 72703 Reutlingen | GERMANY | [www.bosch.com](https://www.bosch.com)
                
                Email: [fixed-term.Akhil.Kumar@de.bosch.com](mailto:fixed-term.Akhil.Kumar@de.bosch.com)
            """, style={'font-family': 'Arial', 'padding': '10px',})
        ], style={'border': '1px solid #007BFF', 'padding': '20px', 'border-radius': '10px', 'backgroundColor': '#FFFFFF'})

@app.callback(
    Output('scatter-matrix', 'figure'),
    [Input('features-dropdown', 'value')]
)
def update_scatter_matrix(selected_features):
    fig = px.scatter_matrix(numerical_data_with_labels, dimensions=selected_features, color='Cluster')
    fig.update_layout(
        title="Scatter Matrix of Selected Features",
        font=dict(family="Arial", size=12, color="black"),
        paper_bgcolor='#F8F9FA',
        plot_bgcolor='#F8F9FA'
    )
    return fig

@app.callback(
    [Output('feature-table', 'figure'), Output('feature-heatmap', 'figure')],
    [Input('feature-dropdown', 'value')]
)
def update_feature_table(selected_features):
    truncated_features = truncate_values(cluster_centroids_features[selected_features])

    table_features = go.Figure(data=[go.Table(
        header=dict(values=["Cluster"] + selected_features,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[truncated_features.index] + [truncated_features[col] for col in selected_features],
                   fill_color='lavender',
                   align='left'))
    ])
    table_features.update_layout(
        title="Original Feature Space Cluster Centroids",
        font=dict(family="Arial", size=12, color="black"),
        paper_bgcolor='#F8F9FA',
        plot_bgcolor='#F8F9FA'
    )

    heatmap_features = px.imshow(cluster_centroids_features[selected_features].values, labels=dict(x="Features", y="Cluster", color="Value"),
                                 x=selected_features, y=cluster_centroids_features.index,
                                 title="Heatmap of Original Feature Space Cluster Centroids")

    return table_features, heatmap_features

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)