import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# Generate sample data
# Replace this with your actual 2D array
data = np.random.rand(10, 3) * 10  # Random data with 100 points and x,y,z coordinates

# Calculate kernel density estimation
kde = gaussian_kde(data.T)

# Define grid for plotting
x_grid, y_grid, z_grid = np.meshgrid(
     np.linspace(data[:, 0].min(), data[:, 0].max(), 50),
     np.linspace(data[:, 1].min(), data[:, 1].max(), 50),
     np.linspace(data[:, 2].min(), data[:, 2].max(), 50)
)

density = kde(np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]))

# Define threshold for transparency
threshold = 0.2  # Adjust as needed

# Normalize density to [0, 1]
density_normalized = (density - density.min()) / (density.max() - density.min())

# Create scatter3d plot
fig = go.Figure(data=go.Scatter3d(
     x=x_grid.ravel(),
     y=y_grid.ravel(),
     z=z_grid.ravel(),
     mode='markers',
     marker=dict(
          size=3,
          color=density_normalized,
          colorscale='Viridis',
          cmin=0,
          cmax=1,
          colorbar=dict(title='Density')
     )
))

# Set opacity based on density
opacity = np.where(density_normalized > threshold, 0.8, density_normalized / threshold * 0.8)
print(len(x_grid.ravel()))
# Add each point individually with its corresponding opacity
for i in range(len(x_grid.ravel())):
     
     fig.add_trace(go.Scatter3d(
          x=[x_grid.ravel()[i]],
          y=[y_grid.ravel()[i]],
          z=[z_grid.ravel()[i]],
          mode='markers',
          marker=dict(
               size=3,
               color=density_normalized[i],
               opacity=opacity[i]
          ),
          showlegend=False
     ))

# Set layout
fig.update_layout(
     scene=dict(
          xaxis_title='X',
          yaxis_title='Y',
          zaxis_title='Z',
     ),
     title='Density Plot'
)

# Save plot as HTML file
# fig.write_html("density_plot_interactive.html")
