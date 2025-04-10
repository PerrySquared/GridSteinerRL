import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
import matplotlib.pyplot as plt


"""
general overflow matrix - gcell overflow matrix for a whole schema
reference overflow matrix - gcell overflow matrix for a small 
standartized area that contains the picked net and is cut out from general overflow matrix 
local overflow matrix - gcell overflow matrix for paths created by algorithm (or RL NN) - 
basically the same thing as path matrix because initially it's empty and is filled with a 
constant value in places where created path is located
"""

OVERFLOW_VALUE = 1

class OverflowWithOverlap():
    def __init__(self, general_overflow_matrix, rows, columns):
        """
        local overflow is the same thing as path due to it being an empty matrix 
        that is getting filled in the same cells as path with some identical 
        values that represent overflow
        """
        self.rows = rows
        self.columns = columns
        self.general_overflow_matrix = general_overflow_matrix
        self.local_overflow_matrix = np.zeros((self.rows, self.columns))
        self.overflow_reference_matrix = np.zeros((self.rows, self.columns))

    def create_overflow_reference_matrix(self, insertion_coords):
        
        # some code to get a cutout of a standertized shape from general overflow matrix in required place
        # rewrite dataser_extractor in a way that gets the origin coordinates of the cutout to later place 
        # the local overflow matrix into the global matrix

        temp_matrix = self.general_overflow_matrix[insertion_coords[0]:insertion_coords[0] + self.rows, insertion_coords[1]:insertion_coords[1] + self.columns]
        
        current_rows, current_cols = temp_matrix.shape

        # Copy the existing matrix into the new matrix to ensure defined dimensions
        self.overflow_reference_matrix[:current_rows, :current_cols] = temp_matrix
        

    def get_manhattan_path(self, start, end):
        x1, y1 = start
        x2, y2 = end
        
        path_length = abs(x1 - x2) + abs(y1 - y2)
        path_sum = 0
        sum_of_normalized_values = 0 

        denominator = np.max(self.overflow_reference_matrix) - np.min(self.overflow_reference_matrix)
        denominator = denominator if denominator > 0 else 1
        
        if y1 <= y2: 
            # Move right first
            # Sum the values moving right
            for x in range(min(x1, x2), max(x1 + 1, x2 + 1)):
                """
                path sum is an overflow for a single L-shape (i.e. path between 
                two points), depends on the refernce from global overflow 
                (a standartized shape cutout of general overflow matrix)
                """
                # swap OVERFLOW_VALUE for overflow_reference_matrix[y1][x] or similar to get a local overflow matrix with values from reference matrix 
                if self.local_overflow_matrix[x][y1] == 0:
                    path_sum += self.overflow_reference_matrix[x][y1]
                    sum_of_normalized_values += (self.overflow_reference_matrix[x][y1] - np.min(self.overflow_reference_matrix))/denominator
                    
                self.local_overflow_matrix[x][y1] = OVERFLOW_VALUE 
            # Sum the values moving down
            for y in range(y1 + 1, y2 + 1):
                if self.local_overflow_matrix[x2][y] == 0:
                    path_sum += self.overflow_reference_matrix[x2][y]
                    sum_of_normalized_values += (self.overflow_reference_matrix[x2][y] - np.min(self.overflow_reference_matrix))/denominator
                    
                self.local_overflow_matrix[x2][y] = OVERFLOW_VALUE

        if y1 > y2:
            # Move down first
            # Sum the values moving down
            for y in range(y2, y1):
                if self.local_overflow_matrix[x2][y] == 0:
                    path_sum += self.overflow_reference_matrix[x2][y]
                    sum_of_normalized_values += (self.overflow_reference_matrix[x2][y] - np.min(self.overflow_reference_matrix))/denominator
                    
                self.local_overflow_matrix[x2][y] = OVERFLOW_VALUE
            # Sum the values moving right
            for x in range(min(x1, x2), max(x1 + 1, x2 + 1)):
                if self.local_overflow_matrix[x][y1] == 0:
                    path_sum += self.overflow_reference_matrix[x][y1]
                    sum_of_normalized_values += (self.overflow_reference_matrix[x][y1] - np.min(self.overflow_reference_matrix))/denominator
                    
                self.local_overflow_matrix[x][y1] = OVERFLOW_VALUE

        return path_sum, sum_of_normalized_values, path_length, self.local_overflow_matrix

    def get_both_manhattan_paths_sums(self, matrix, start, finish):
        path_1, matrix_1 = self.get_manhattan_path(matrix, start, finish)
        path_2, matrix_2 = self.get_manhattan_path(matrix, finish, start)

        if path_1 > path_2:
            smaller_overflow = path_2
            smaller_overflow_matrix = matrix_2
        else:
            smaller_overflow = path_1
            smaller_overflow_matrix = matrix_1

        return smaller_overflow, smaller_overflow_matrix


class OverflowWithOverlap3D():
    def __init__(self, general_overflow_matrix, rows, columns, layers):
        """
        3D version that handles volumetric matrices with depth/layers
        local overflow is a 3D matrix that gets filled in the same cells as path
        """
        self.rows = rows
        self.columns = columns
        self.layers = layers
        self.general_overflow_matrix = general_overflow_matrix
        self.local_overflow_matrix = np.zeros((self.rows, self.columns, self.layers))
        self.overflow_reference_matrix = np.zeros((self.rows, self.columns, self.layers))
        self.path_only_matrix = np.zeros((self.rows, self.columns, self.layers))

    def create_overflow_reference_matrix(self, insertion_coords):
        # Extract a 3D submatrix from the general overflow matrix
        x_start, y_start, z_start = insertion_coords
        
        temp_matrix = self.general_overflow_matrix[
            x_start:x_start + self.rows, 
            y_start:y_start + self.columns,
            z_start:z_start + self.layers 
        ]
        
        current_rows, current_cols, current_layers = temp_matrix.shape

        # Copy the existing matrix into the new matrix to ensure defined dimensions
        self.overflow_reference_matrix[:current_rows, :current_cols, :current_layers] = temp_matrix
        
    def get_manhattan_path(self, start, end, path_type=0, target_layer=None):
        """
        Calculate Manhattan path between two points with selectable path type and optional layer constraint
        Excludes start and end points from the path calculations
        
        Parameters:
        - start: (x1, y1, z1) starting point
        - end: (x2, y2, z2) ending point
        - path_type: Integer 0-1 selecting the path pattern on target layer
        - target_layer: Optional layer (z-coordinate) that the path must pass through
        
        Returns: (path_sum, avg_normalized_value, path_length, local_overflow_matrix)
        """
        
        x1, y1, z1 = start
        x2, y2, z2 = end

        # Setup normalization factors
        ref_min = np.min(self.overflow_reference_matrix)
        ref_max = np.max(self.overflow_reference_matrix)
        denominator = max(ref_max - ref_min, 1)
        overflow_reference_matrix_normalized = (self.overflow_reference_matrix - ref_min) / denominator
        
        # Calculate target layer Z or use end Z
        z_target = target_layer if target_layer is not None else z2
        
        # Mark start and end as visited without processing
        self.local_overflow_matrix[x1, y1, z1] = OVERFLOW_VALUE
        self.local_overflow_matrix[x2, y2, z2] = OVERFLOW_VALUE
        
        # Initialize counters
        path_sum = 0
        sum_of_normalized_values = 0
        path_length = 0
        
        # Helper function to process a point on the fly
        def process_point(x, y, z):
            nonlocal path_sum, sum_of_normalized_values, path_length
            if not (x == x2 and y == y2 and z == z2) and not (x == x1 and y == y1 and z == z1) and self.local_overflow_matrix[x, y, z] == 0:
                # print(self.overflow_reference_matrix[x, y, z])
                path_sum += self.overflow_reference_matrix[x, y, z]
                sum_of_normalized_values += overflow_reference_matrix_normalized[x, y, z]
                path_length += 1
            self.local_overflow_matrix[x, y, z] = OVERFLOW_VALUE
        
        # 1. First segment: vertical move from start to target layer
        x_current, y_current = x1, y1
        z_step = 1 if z_target > z1 else -1
        if z1 != z_target:
            for z in range(z1 + z_step, z_target + z_step, z_step):
                if not (x_current == x2 and y_current == y2 and z == z2):
                    process_point(x_current, y_current, z)
        
        # 2. Movement on target layer
        if path_type == 0:  # X → Y
            # Move in X direction first
            x_step = 1 if x2 > x_current else -1
            if x_current != x2:
                for x in range(x_current + x_step, x2 + x_step, x_step):
                    if not (x == x2 and y_current == y2 and z_target == z2):
                        process_point(x, y_current, z_target)
            
            # Then move in Y direction
            x_current = x2
            y_step = 1 if y2 > y_current else -1
            if y_current != y2:
                for y in range(y_current + y_step, y2 + y_step, y_step):
                    if not (x_current == x2 and y == y2 and z_target == z2):
                        process_point(x_current, y, z_target)
        else:  # Y → X (path_type = 1)
            # Move in Y direction first
            y_step = 1 if y2 > y_current else -1
            if y_current != y2:
                for y in range(y_current + y_step, y2 + y_step, y_step):
                    if not (x_current == x2 and y == y2 and z_target == z2):
                        process_point(x_current, y, z_target)
            
            # Then move in X direction
            y_current = y2
            x_step = 1 if x2 > x_current else -1
            if x_current != x2:
                for x in range(x_current + x_step, x2 + x_step, x_step):
                    if not (x == x2 and y_current == y2 and z_target == z2):
                        process_point(x, y_current, z_target)
        
        # 3. Final segment: vertical move from target layer to end
        x_current, y_current = x2, y2
        if z_target != z2:
            z_step = 1 if z2 > z_target else -1
            for z in range(z_target + z_step, z2, z_step):
                if not (x_current == x2 and y_current == y2 and z == z2):
                    process_point(x_current, y_current, z)
        
        # Calculate average normalized value
        avg_normalized_value = sum_of_normalized_values / path_length if path_length > 0 else 0
        
        # print(path_sum, avg_normalized_value, path_length, "\n-------------------\n")
        return path_sum, avg_normalized_value, path_length, self.local_overflow_matrix
    
        




# general_overflow_matrix = np.zeros([16, 16]) 
# local_overflow_matrix = np.zeros([16, 16]) 

# c = OverflowWithOverlap(general_overflow_matrix, local_overflow_matrix, 16, 16)

# print(c.get_manhattan_path((3,5), (8,9)))
# print(c.get_manhattan_path((8,9), (10,3)))
# print(c.get_manhattan_path((10,3), (6,2)))
# print(c.get_manhattan_path((6,2), (1,7)))
# print(c.get_manhattan_path((1,7), (6,0)))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define the OVERFLOW_VALUE constant
OVERFLOW_VALUE = 1

def test_overflow_with_overlap_3d_plotly():
    # Create a 3D general overflow matrix
    size = 16
    LAYERS = 6
    np.random.seed(42)
    general_overflow_matrix = np.random.randint(0, 100, size=(size, size, LAYERS))
    
    print(f"Generated matrix range: min={np.min(general_overflow_matrix)}, " 
          f"max={np.max(general_overflow_matrix)}, "
          f"unique values: {len(np.unique(general_overflow_matrix))}")
    
    # Initialize the OverflowWithOverlap3D instance
    c = OverflowWithOverlap3D(general_overflow_matrix, size, size, LAYERS)
    c.create_overflow_reference_matrix((0, 0, 0))
    
    print(f"Reference matrix range: min={np.min(c.overflow_reference_matrix)}, " 
          f"max={np.max(c.overflow_reference_matrix)}")
    
    # Reset the local overflow matrix
    c.local_overflow_matrix = np.zeros((size, size, LAYERS))
    
    # Create path points (x, y, z)
    path_points = [
        (5, 5, 1),    # Starting point
        (8, 7, 3),    # End of path 1, start of path 2
        (10, 4, 5),   # End of path 2, start of path 3
        (6, 2, 3),    # End of path 3, start of path 4
        (3, 6, 4),    # End of path 4, start of path 5
        (7, 6, 7),   # End of path 5
        (7, 7, 7)
    ]
    
    # Configure test parameters
    # With target layer, path_type is limited to 0 or 1:
    # 0: X→Y on target layer, 1: Y→X on target layer
    target_layers = [2, 7, 4, 1, 7, 7]
    path_types = [1, 1, 1, 0, 1, 0]  # Alternating X→Y and Y→X on target layer
    
    # Collect path information
    all_path_points = []
    
    # Test each path segment
    for i in range(len(path_points)-1):
        start = path_points[i]
        end = path_points[i+1]
        path_type = path_types[i]
        target_layer = target_layers[i]
        
        print(f"\nPath {i+1}: from {start} to {end} with path type {path_type} and target layer {target_layer}:")
        path_sum, norm_avg, length, _ = c.get_manhattan_path(start, end, path_type, target_layer)
        print(f"Path sum: {path_sum:.2f}, Normalized avg: {norm_avg:.2f}, Length: {length}")
        
        # Generate visualization path points
        path_segment = generate_manhattan_path_points(start, end, path_type, target_layer)
        all_path_points.extend(path_segment)
        
        # Verify points at target layer
        if target_layer is not None:
            layer_slice = c.local_overflow_matrix[:, :, target_layer]
            has_points_at_target = np.any(layer_slice == OVERFLOW_VALUE)
            print(f"Path includes points at target layer {target_layer}: {has_points_at_target}")
            if has_points_at_target:
                target_coords = np.where(layer_slice == OVERFLOW_VALUE)
                # print(f"  Coordinates at target layer: {list(zip(target_coords[0], target_coords[1]))}")
    
    # Visualize with Plotly
    visualize_3d_plotly(c.local_overflow_matrix, c.overflow_reference_matrix, 
                      path_points, all_path_points, path_types, target_layers)
    
    # Output summary
    print("\nSummary of target layer paths:")
    for i in range(len(path_points)-1):
        start = path_points[i]
        end = path_points[i+1]
        path_type = path_types[i]
        target = target_layers[i]
        
        xy_pattern = "X→Y" if path_type == 0 else "Y→X"
        print(f"Path {i+1}: {start[2]}→{target}→{end[2]} Z, with {xy_pattern} on layer {target}")

def generate_manhattan_path_points(start, end, path_type=0, target_layer=None):
    """Generate all points along a Manhattan path with target layer constraint"""
    points = [start]
    x1, y1, z1 = start
    x2, y2, z2 = end
    
    # If no target layer specified
    if target_layer is None:
        # Define the axis order for different path types (X=0, Y=1, Z=2)
        axis_orders = [
            [2, 0, 1],  # Z → X → Y (path_type 0)
            [2, 1, 0],  # Z → Y → X (path_type 1)
            [0, 2, 1],  # X → Z → Y (path_type 2)
            [0, 1, 2],  # X → Y → Z (path_type 3)
            [1, 2, 0],  # Y → Z → X (path_type 4)
            [1, 0, 2]   # Y → X → Z (path_type 5)
        ]
        
        # Ensure valid path_type
        path_type = min(max(0, path_type), 5)
        
        # Follow standard Manhattan path
        current = list(start)
        for axis in axis_orders[path_type]:
            # Determine direction and range
            start_val = current[axis]
            end_val = end[axis]
            step = 1 if end_val >= start_val else -1
            
            # Move along this axis
            for val in range(start_val + step, end_val + step, step):
                current[axis] = val
                points.append(tuple(current))
        return points
    
    # With target layer, we follow a 3-phase approach:
    # 1. Move Z from start to target layer
    # 2. Move X and Y on target layer according to path_type (0: X→Y, 1: Y→X)
    # 3. Move Z from target layer to end
    
    # PHASE 1: Move Z from start to target layer
    current = list(start)
    z_step = 1 if target_layer >= z1 else -1
    for z in range(z1 + z_step, target_layer + z_step, z_step):
        current[2] = z
        points.append(tuple(current))
    
    # PHASE 2: Move X and Y on target layer
    if path_type == 0:  # X → Y
        # Move X
        x_step = 1 if x2 >= x1 else -1
        for x in range(x1 + x_step, x2 + x_step, x_step):
            current[0] = x
            points.append(tuple(current))
        
        # Move Y
        y_step = 1 if y2 >= y1 else -1
        for y in range(y1 + y_step, y2 + y_step, y_step):
            current[1] = y
            points.append(tuple(current))
    else:  # Y → X
        # Move Y
        y_step = 1 if y2 >= y1 else -1
        for y in range(y1 + y_step, y2 + y_step, y_step):
            current[1] = y
            points.append(tuple(current))
        
        # Move X
        x_step = 1 if x2 >= x1 else -1
        for x in range(x1 + x_step, x2 + x_step, x_step):
            current[0] = x
            points.append(tuple(current))
    
    # PHASE 3: Move Z from target layer to end
    z_step = 1 if z2 >= target_layer else -1
    for z in range(target_layer + z_step, z2 + z_step, z_step):
        current[2] = z
        points.append(tuple(current))
    
    return points


# Make sure visualize_3d_plotly accepts and uses target_layers parameter
def visualize_3d_plotly(local_matrix, reference_matrix, junction_points, all_path_points, path_types, target_layers=None):
    """Create adaptive interactive 3D visualization with reference values on path points"""
    
    # If target_layers not provided, use None for all segments
    if target_layers is None:
        target_layers = [None] * (len(junction_points) - 1)
    
    # Create a figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('Local Overflow Matrix (Paths)', 'Overflow Reference Matrix'),
        horizontal_spacing=0.03
    )
    
    # Add path points to the first subplot
    path_x = [p[0] for p in all_path_points]
    path_y = [p[1] for p in all_path_points]
    path_z = [p[2] for p in all_path_points]
    ref_values = [reference_matrix[x, y, z] for x, y, z in all_path_points]
    
    fig.add_trace(
        go.Scatter3d(
            x=path_x, y=path_y, z=path_z,
            mode='markers',
            marker=dict(
                size=5,
                color=ref_values,
                colorscale='Viridis',
                opacity=0.9,
                colorbar=dict(title='Reference Value', x=0.45),
                showscale=True
            ),
            name='Path Points',
            hoverinfo='text',
            text=[f"Point: ({x}, {y}, {z})<br>Reference Value: {v:.4f}" 
                  for x, y, z, v in zip(path_x, path_y, path_z, ref_values)]
        ),
        row=1, col=1
    )
    
    # Add junction points
    junc_x = [p[0] for p in junction_points]
    junc_y = [p[1] for p in junction_points]
    junc_z = [p[2] for p in junction_points]
    junc_ref_values = [reference_matrix[x, y, z] for x, y, z in junction_points]
    
    fig.add_trace(
        go.Scatter3d(
            x=junc_x, y=junc_y, z=junc_z,
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                symbol='diamond',
                line=dict(color='black', width=1)
            ),
            name='Junction Points',
            hoverinfo='text',
            text=[f"Junction {i+1}: ({x}, {y}, {z})<br>Reference Value: {v:.4f}" 
                  for i, (x, y, z, v) in enumerate(zip(junc_x, junc_y, junc_z, junc_ref_values))]
        ),
        row=1, col=1
    )
    
    # Add path segments with correct routing through target layers
    for i in range(len(junction_points)-1):
        start = junction_points[i]
        end = junction_points[i+1]
        path_type = path_types[i]
        target_layer = target_layers[i]
        
        # Generate path points with target layer constraint
        path_segment = generate_manhattan_path_points(start, end, path_type, target_layer)
        
        seg_x = [p[0] for p in path_segment]
        seg_y = [p[1] for p in path_segment]
        seg_z = [p[2] for p in path_segment]
        
        # Get reference values
        seg_ref_values = [reference_matrix[x, y, z] for x, y, z in path_segment]
        seg_avg_value = sum(seg_ref_values) / len(seg_ref_values) if seg_ref_values else 0
        
        # Add target layer info to hover text
        target_info = f"<br>Target Layer: {target_layer}" if target_layer is not None else ""
        
        fig.add_trace(
            go.Scatter3d(
                x=seg_x, y=seg_y, z=seg_z,
                mode='lines',
                line=dict(color='green', width=6),
                name=f'Path {i+1}',
                hoverinfo='text',
                text=f"Path {i+1}: {start} to {end}{target_info}<br>Avg Value: {seg_avg_value:.4f}"
            ),
            row=1, col=1
        )
    
    # Add to second subplot (reference matrix visualization)
    # Junction points
    fig.add_trace(
        go.Scatter3d(
            x=junc_x, y=junc_y, z=junc_z,
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                symbol='diamond'
            ),
            name='Junctions',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Path points
    fig.add_trace(
        go.Scatter3d(
            x=path_x, y=path_y, z=path_z,
            mode='markers',
            marker=dict(
                size=4,
                color=ref_values,
                colorscale='Viridis',
                opacity=0.8,
                showscale=False
            ),
            name='Path Points',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add path lines to second subplot
    for i in range(len(junction_points)-1):
        start = junction_points[i]
        end = junction_points[i+1]
        path_type = path_types[i]
        target_layer = target_layers[i]
        
        path_segment = generate_manhattan_path_points(start, end, path_type, target_layer)
        
        seg_x = [p[0] for p in path_segment]
        seg_y = [p[1] for p in path_segment]
        seg_z = [p[2] for p in path_segment]
        
        fig.add_trace(
            go.Scatter3d(
                x=seg_x, y=seg_y, z=seg_z,
                mode='lines',
                line=dict(color='green', width=3, dash='dash'),
                opacity=0.7,
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Visualize reference matrix values (background)
    ref_x, ref_y, ref_z = np.indices(reference_matrix.shape)
    ref_x = ref_x.flatten()
    ref_y = ref_y.flatten()
    ref_z = ref_z.flatten()
    values = reference_matrix.flatten()
    
    # Only show significant values
    threshold = np.percentile(values, 75)
    mask = values > threshold
    
    fig.add_trace(
        go.Scatter3d(
            x=ref_x[mask], y=ref_y[mask], z=ref_z[mask],
            mode='markers',
            marker=dict(
                size=2,
                color=values[mask],
                colorscale='Viridis',
                opacity=0.3,
                colorbar=dict(title='Reference Values', x=0.98),
                showscale=True
            ),
            name='Background Values',
            hoverinfo='skip'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': '3D Visualization of Overflow Paths with Target Layers',
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        autosize=True,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    # Update scenes for consistent axis ranges
    axis_range = dict(range=[0, reference_matrix.shape[0]])  # Changed from shape[1] to shape[0]
    
    for col in [1, 2]:
        fig.update_scenes(
            xaxis=dict(title='X Axis', **axis_range),
            yaxis=dict(title='Y Axis', **axis_range),
            zaxis=dict(title='Z Axis', range=[0, reference_matrix.shape[2]]),  # Specific Z range
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            row=1, col=col
        )
    
    # Show the figure
    fig.show(config={
        'responsive': True,
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'displaylogo': False
    })

# Run the test with Plotly visualization
if __name__ == "__main__":
    test_overflow_with_overlap_3d_plotly()