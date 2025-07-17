import gymnasium as gym 
import gym_examples 
import numpy as np 
import matplotlib.pyplot as plt 
import torch
import time
import plotly.graph_objects as go
import copy
import h5py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.callbacks import EvalCallback 
from stable_baselines3 import PPO 
from stable_baselines3.common.evaluation import evaluate_policy 
from alg_router.boxrouter_gr import FullGlobalRouter
from alg_router.dataset_extractor import get_coords_dataset_3d

f = h5py.File('./gym_examples/envs/utils/dataset_3d_blabla.h5', 'r')

OUT_GENERAL_OVERFLOW = np.zeros((217, 217, 6), dtype=np.float64)
 
# List of model paths to iterate through
model_paths = [
    "./final_models/ppo_model_2pin_11.04.zip",
    "./final_models/ppo_model_3pin_11.04.zip",
    "./final_models/ppo_model_4pin_11.04.zip",
    "./final_models/ppo_model_5pin_10.04.zip"
    # Add more models as needed
]

# For recording results if needed
results = {}

def visualize_overflow_matrix(overflow_matrix):
    # Create figure with 2x3 subplots for the 6 layers
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}],
               [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}]],
        subplot_titles=[f"Layer {i+1}" for i in range(6)],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    # Add heatmaps for each layer
    for i in range(6):
        row, col = i // 3 + 1, i % 3 + 1
        heatmap = go.Heatmap(
            z=overflow_matrix[:, :, i],
            colorscale='Viridis',
            showscale=(i == 5),  # Only show colorbar for the last layer
        )
        fig.add_trace(heatmap, row=row, col=col)
    
    # Update layout for the 2D visualization
    fig.update_layout(
        title_text="Routing Congestion Visualization by Layer",
        height=800,
        width=1200,
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(
            title="Congestion",
            x=1.02,  # Position colorbar at right edge
            xanchor="left"
        )
    )
    
    # Create a 3D scatter visualization with fixed-size dots
    # First, create a downsampled version to avoid too many points
    # Only include non-zero congestion points
    points = []
    for i in range(overflow_matrix.shape[0]):
        for j in range(overflow_matrix.shape[1]):
            for k in range(overflow_matrix.shape[2]):
                if overflow_matrix[i, j, k] > 0:
                    points.append((i, j, k, overflow_matrix[i, j, k]))
    
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    values = [p[3] for p in points]
    
    # Create 3D scatter plot
    fig3d = go.Figure()
    
    # Add scatter points with fixed size 
    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,            # Fixed size - no variation
            sizemode='diameter',  # Ensure consistent size
            color=values,      # Color varies by congestion value
            colorscale='Viridis',
            colorbar=dict(
                title="Congestion",
                x=0.95         # Position colorbar at right edge
            )
        ),
        hovertemplate='X: %{x}<br>Y: %{y}<br>Layer: %{z}<br>Value: %{marker.color:.2f}<extra></extra>'
    )
    
    fig3d.add_trace(scatter)
    
    # Update 3D layout
    fig3d.update_layout(
        title='3D Routing Congestion Visualization',
        width=900,
        height=800,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Layer',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig, fig3d
# Function to extract pin number from model name
def extract_pin_number(model_name):
    # Extract the pin number from the model name (e.g., "ppo_model_2pin_11.04.zip" → 2)
    import re
    match = re.search(r'(\d+)pin', model_name)
    if match:
        return int(match.group(1))
    else:
        print("error: no such pin count")

def write_to_global_overflow(insertion_coords, origin_shape, rows, columns, layers, route):
    upto_row = min(insertion_coords[0] + rows, origin_shape[0])
    upto_column = min(insertion_coords[1] + columns, origin_shape[1])
    upto_z = min(insertion_coords[2] + layers, origin_shape[2])

    limit_row = origin_shape[0] - insertion_coords[0]
    limit_column = origin_shape[1] - insertion_coords[1]
    limit_z = origin_shape[2] - insertion_coords[2]
    
    OUT_GENERAL_OVERFLOW[
        insertion_coords[0]:upto_row, 
        insertion_coords[1]:upto_column,
        insertion_coords[2]:upto_z
    ] += route[0:limit_row, 0:limit_column, 0:limit_z]

for model_path in model_paths:
    # Determine if we're at the last iteration
    model_name = model_path.split("/")[-1]
    
    pin_count = extract_pin_number(model_name)
    
    if model_name not in results:
        results[model_name] = {}
    
    # Print configuration info
    print(f"\nTesting model: {model_name} with ppo_approach: {True}, pins: {pin_count}")
    
    # Create environment with the correct pin count
    vec_env = make_vec_env("gym_examples/GridWorld-v0",
                          env_kwargs={'ppo_aproach': True,
                                     'pins': pin_count,
                                     'general_overflow_matrix': OUT_GENERAL_OVERFLOW,
                                     'file': f}
                          ) 

    model = PPO.load(model_path, env=vec_env)
    
    obs = vec_env.reset()
    lstm_states = None
    num_envs = 1
    episode_starts = np.ones((num_envs,), dtype=bool)
    
    print("RL Routing starts")
    
    start = time.time()
    nets_count = 0
    failed_count = 0
    
    insertion_coords = []

    while(type(insertion_coords) is not bool):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        episode_starts = dones
        
        failed = info[0]['TimeLimit.truncated']
        
        if len(info[0]) > 5:
            
            path_matrix = info[0]['terminal_observation']['target_matrix'] * 2
            insertion_coords = info[0]['insertion_coords']
            origin_shape = info[0]['origin_shape']
            # targets_coords = info[0]['terminal_observation']['target_list']
            # local_overflow_matrix = info[0]['local_overflow_matrix']
            # reference_overflow_matrix = info[0]['reference_overflow_matrix']

            if not failed and type(insertion_coords) is not bool: # picked insertion coords, but other options could be used if they also return False on grid_world 
                nets_count += 1  
                # write_to_global_overflow(insertion_coords, origin_shape, 32, 32, 6, path_matrix)
        
            if failed and type(insertion_coords) is not bool:
                failed_count += 1

                # print(f"Failed at net {nets_count}, total fails: {failed_count}")
                
                # router = FullGlobalRouter(local_overflow_matrix, reference_overflow_matrix, alpha=1.0)
                # route = router.global_routing()

                # write_to_global_overflow(insertion_coords, origin_shape, 32, 32, 6, route)
                # use guranteed solution for the failed route and insert it into the global matrix
    
    end = time.time()
    elapsed = end - start
    
    results[model_name][f"ppo_approach_{True}"] = {
        "time": elapsed,
        "nets_routed": nets_count,
        "failed_count": failed_count
    }
    
    print(f"Completed in {elapsed:.2f} seconds with {failed_count} failures")
    
    vec_env.close()



print("\nResults Summary:")
for model, approaches_data in results.items():
    print(f"\nModel: {model}")
    for approach, data in approaches_data.items():
        print(f"  {approach}: Time={data['time']:.2f}s, Nets={data['nets_routed']}, Failures={data['failed_count']}")
        

# FINISH OFF WITH RUNNING A REAL ROUTER ON EVERYTHING THAT DIDNT MEET THE CRITERIA OF THE MODEL
# i.e. x,y >32, amount of pins > 5 etc 
# add everything to global matrix

print("ALG Routing starts")


def get_coordinates_3d(matrix):
    coordinates = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                if matrix[i, j, k] == 1:
                    coordinates.append(np.array([i, j, k]))
    return coordinates
  

group_keys = list(f.keys())
extra_counter = 0
LAYERS = 6
    
start = time.time()

for i in range(len(group_keys)):
    key = group_keys[i]
    group = f[key]

    matrix = group[key][:]  
    
    net_name = group.attrs.get("net_name", "Unknown")
    insertion_coords = group.attrs.get("insertion_coords", (0, 0, 0))
    origin_shape = group.attrs.get("origin_shape", (0, 0, 0))
    
    
    target_count_on_matrix = np.count_nonzero(matrix == 1)
    
    if (matrix.shape[0] >= 32 or 
        matrix.shape[1] >= 32 or 
        target_count_on_matrix > 4): # take into account everything that's beyond model's limitations
        
        extra_counter += 1
        
        insertion_coords = np.append(insertion_coords, 0)
        temp_target_locations_copy, net_name, insertion_coords, origin_shape, matrix_shape, found_instance_index = get_coordinates_3d(matrix), net_name, insertion_coords, origin_shape, matrix.shape, i

        x_start, y_start, z_start = insertion_coords
        rows, columns, layers = matrix_shape

        temp_matrix = OUT_GENERAL_OVERFLOW[
            x_start:x_start + rows, 
            y_start:y_start + columns,
            z_start:z_start + layers 
        ]

        current_rows, current_cols, current_layers = temp_matrix.shape

        # Copy the existing matrix into the new matrix to ensure defined dimensions
        overflow_reference_matrix = np.zeros((matrix_shape))
        overflow_reference_matrix[:current_rows, :current_cols, :current_layers] = temp_matrix

        local_overflow_matrix = np.zeros((matrix_shape))

        for _target_location in temp_target_locations_copy:
            if _target_location[0] != -1 and _target_location[1] != -1 and _target_location[2] != -1:
                # print(_target_location[0], _target_location[1], _target_location[2])
                local_overflow_matrix[_target_location[0], _target_location[1], _target_location[2]] = 1

            
        alpha = 1.0

        router = FullGlobalRouter(local_overflow_matrix, overflow_reference_matrix, alpha)
        route = router.global_routing()
        
        write_to_global_overflow(insertion_coords, origin_shape, rows, columns, layers, route)
        
end = time.time()

print('alg time = ', end-start)
print('algo nets = ', extra_counter)

print("\nGenerating visualizations...")
print("mean = ", np.mean(OUT_GENERAL_OVERFLOW))
print("avg = ", np.average(OUT_GENERAL_OVERFLOW))
print("total = ", np.sum(OUT_GENERAL_OVERFLOW))
print("min = ", np.min(OUT_GENERAL_OVERFLOW), "max = ", np.max(OUT_GENERAL_OVERFLOW))



# Add this after the results summary
layer_fig, scatter_fig = visualize_overflow_matrix(OUT_GENERAL_OVERFLOW)
layer_fig.show()
scatter_fig.show()

# Optional: Save the figures
# layer_fig.write_html("layer_visualization.html")
# scatter_fig.write_html("3d_scatter_visualization.html")
