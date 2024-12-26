import pandas as pd
import os
import umap
import plotly.express as px
import torch
from sklearn.preprocessing import StandardScaler

# Directory path where CSV files are stored
data_dir = "/Users/akhilkumar/Downloads/saska_enc"

# Create an empty list to store dataframes and file names
all_dataframes = []
file_names = []  # To keep track of file names for labeling

# Loop through all CSV files in the directory and load them into a list
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(data_dir, filename)
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(filepath)
            # Append the DataFrame to the list
            all_dataframes.append(df)
            # Extract last 5 characters of the filename (excluding '.csv') and store
            short_name = filename[-9:-4]  # Last 5 characters before ".csv"
            file_names.extend([short_name] * len(df))  # Repeat short name for each row in the CSV
            print(f"Successfully loaded: {filename}")
        except pd.errors.ParserError as e:
            print(f"Error parsing {filename}: {e}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Concatenate all DataFrames into a single DataFrame
if all_dataframes:
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Total number of rows combined: {len(combined_df)}")

    # Select only numerical columns (skip non-numeric columns like file names)
    numerical_data = combined_df.select_dtypes(include=['float64', 'int64'])

    # Save the numerical data to a CSV file for later use
    numerical_data.to_csv("/Users/akhilkumar/Downloads/UMAP/numerical_data.csv", index=False)

    # Step 1: Standardize the data (scaling)
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numerical_data)

    # Convert standardized data to a PyTorch tensor and move to MPS device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    standardized_data_tensor = torch.tensor(standardized_data, device=device, dtype=torch.float32)

    # Step 2: Apply UMAP for dimensionality reduction (to 3D)
    umap_model = umap.UMAP(n_components=3, random_state=42)
    umap_data = umap_model.fit_transform(standardized_data_tensor.cpu().numpy())

    # Convert UMAP results to a DataFrame
    umap_df = pd.DataFrame(umap_data, columns=['UMAP 1', 'UMAP 2', 'UMAP 3'])

    # Save the UMAP DataFrame to a CSV file for later use
    umap_df.to_csv("/Users/akhilkumar/Downloads/UMAP/umap_results.csv", index=False)

    # Sample a subset of the UMAP data for plotting to reduce computational load
    sampled_umap_df = umap_df.sample(n=1000, random_state=42)  # Adjust the sample size as needed

    # Create a 3D scatter plot with Plotly
    fig = px.scatter_3d(sampled_umap_df, x='UMAP 1', y='UMAP 2', z='UMAP 3', title="3D UMAP Projection")

    # Show the plot
    fig.show()

else:
    print("No CSV files found or successfully loaded in the specified directory.")