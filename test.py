import numpy as np
import plotly.graph_objs as go
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# Generate sample data (you can replace this with your own data)
n_samples = 1000
n_features = 3  # x, y, z
noise_level = 0.1
random_state = 42

X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=5, cluster_std=1.0, random_state=random_state)
# Adding noise to the data
X = np.append(X, np.random.randn(int(n_samples*noise_level), n_features), axis=0)

print(np.shape(X))
# Apply DBSCAN clustering algorithm
dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan.fit(X)
labels = dbscan.labels_

# Create a scatter plot
scatter = go.Scatter3d(
    x=X[:, 0],
    y=X[:, 1],
    z=X[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=labels,
        colorscale='Viridis',
        opacity=0.8
    )
)

# Plot layout
layout = go.Layout(
    title='3D Scatter Plot with DBSCAN Clusters',
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
    )
)

# Create the figure
fig = go.Figure(data=[scatter], layout=layout)

# Show the interactive plot
fig.show()

# Save the plot as HTML
fig.write_html("3d_scatter_plot.html", auto_open=True)
