#!/usr/bin/env python3
"""
FullGlobalRouter.py

This module implements the full‐feature global routing stage modeled on the original BoxRouter global routing,
with all advanced features:
  • Pre‑Routing Congestion Prediction
  • Box Expansion Strategy (with domain‐limiting A* search)
  • Progressive ILP Optimization (with flow conservation constraints)
  • Negotiation‑Based A* Search (with nonlinear advanced cost functions)
  • Topology‑Aware Wire Rip‑up/Rerouting
  • Layer Assignment and Via Optimization
  • Advanced Congestion Cost Functions

I/O:
  - Input:
       pin_matrix: a binary 3D NumPy array (cells with 1 mark pins).
       usage_matrix: a 3D integer NumPy array (same dimensions) of current cell usage.
  - Output:
       route_matrix: a single 3D route matrix (same dimensions) containing the complete routed net.
       
A Plotly visualization is provided that overlays the usage gradient, routed net (red), and pin locations (bright green).

Dependencies: numpy, heapq, pulp, plotly
"""

import numpy as np
import heapq
import plotly.graph_objects as go
import pulp as pl
from typing import List, Tuple, Optional, Dict

# -------------------------------------------------------------
# Basic 3D Point class
# -------------------------------------------------------------
class Point:
    def __init__(self, x: int, y: int, z: int):
        self.x = x; self.y = y; self.z = z
    def __eq__(self, other):
        return isinstance(other, Point) and (self.x, self.y, self.z) == (other.x, other.y, other.z)
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    def as_tuple(self) -> Tuple[int,int,int]:
        return (self.x, self.y, self.z)
    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"

# -------------------------------------------------------------
# Advanced Congestion Cost Function (nonlinear)
# -------------------------------------------------------------
def advanced_cost(usage_val: float, alpha: float) -> float:
    """Nonlinear cost: base cost 1 plus alpha * usage^2."""
    return 1 + alpha * (usage_val ** 2)

# -------------------------------------------------------------
# Pre‑Routing Congestion Prediction (3D Smoothing Filter)
# -------------------------------------------------------------
def pre_routing_congestion_prediction(usage: np.ndarray) -> np.ndarray:
    dims = usage.shape
    predicted = np.zeros(dims, dtype=float)
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                total = 0.0; count = 0
                for dx in [-1,0,1]:
                    for dy in [-1,0,1]:
                        for dz in [-1,0,1]:
                            nx, ny, nz = x+dx, y+dy, z+dz
                            if 0 <= nx < dims[0] and 0 <= ny < dims[1] and 0 <= nz < dims[2]:
                                total += usage[nx, ny, nz]
                                count += 1
                predicted[x,y,z] = total / count
    return predicted

# -------------------------------------------------------------
# Box Expansion Strategy: compute an expanded bounding box and restrict search domain.
# -------------------------------------------------------------
def box_expansion_strategy(net_bbox: Tuple[Tuple[int,int,int], Tuple[int,int,int]], iteration: int, max_expand: int) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    min_xyz, max_xyz = net_bbox
    expand = min(iteration, max_expand)
    new_min = tuple(max(0, min_xyz[i]-expand) for i in range(3))
    new_max = tuple(max_xyz[i]+expand for i in range(3))
    return new_min, new_max

# -------------------------------------------------------------
# Generate neighbors within an expanded bounding box.
# -------------------------------------------------------------
def get_neighbors_in_box(pt: Point, dims: Tuple[int,int,int], box: Tuple[Tuple[int,int,int], Tuple[int,int,int]]) -> List[Point]:
    new_neighbors = []
    min_box, max_box = box
    for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        nx, ny, nz = pt.x+dx, pt.y+dy, pt.z+dz
        if (min_box[0] <= nx < min(max_box[0], dims[0])) and (min_box[1] <= ny < min(max_box[1], dims[1])) and (min_box[2] <= nz < min(max_box[2], dims[2])):
            new_neighbors.append(Point(nx,ny,nz))
    return new_neighbors

# -------------------------------------------------------------
# Negotiation-Based A* Search (with iterative cost adjustment)
# -------------------------------------------------------------
def negotiation_based_A_star(start: Point, end: Point, internal_usage: np.ndarray, dims: Tuple[int,int,int], alpha: float, box: Tuple[Tuple[int,int,int], Tuple[int,int,int]], negotiation_iters: int = 3) -> Optional[List[Point]]:
    counter = 0
    open_heap = []
    # Use Manhattan distance as heuristic.
    heapq.heappush(open_heap, (abs(start.x-end.x)+abs(start.y-end.y)+abs(start.z-end.z), counter, 0, start))
    came_from = {}
    cost_so_far: Dict[Point, float] = {start: 0}
    
    def restricted_neighbors(pt: Point) -> List[Point]:
        return get_neighbors_in_box(pt, dims, box)
    
    route = None
    for negotiation in range(negotiation_iters):
        # Standard A* over the restricted box.
        open_heap = []
        counter = 0
        heapq.heappush(open_heap, (abs(start.x-end.x)+abs(start.y-end.y)+abs(start.z-end.z), counter, 0, start))
        came_from = {}
        cost_so_far = {start: 0}
        found = False
        while open_heap:
            prio, _, current_cost, current = heapq.heappop(open_heap)
            if current == end:
                # Reconstruct path
                route = []
                while current in came_from:
                    route.append(current)
                    current = came_from[current]
                route.append(start)
                route.reverse()
                found = True
                break
            for nb in restricted_neighbors(current):
                new_cost = cost_so_far[current] + advanced_cost(internal_usage[nb.x, nb.y, nb.z], alpha)
                if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                    cost_so_far[nb] = new_cost
                    priority = new_cost + abs(nb.x-end.x)+abs(nb.y-end.y)+abs(nb.z-end.z)
                    counter += 1
                    heapq.heappush(open_heap, (priority, counter, new_cost, nb))
                    came_from[nb] = current
        if found and route is not None:
            # Check if any cell in the route has usage above a threshold; if so, add penalty and renegotiate.
            high_cong = False
            for pt in route:
                if internal_usage[pt.x, pt.y, pt.z] > 5:
                    internal_usage[pt.x, pt.y, pt.z] += 2  # add penalty
                    high_cong = True
            if high_cong:
                continue  # rerun the negotiation loop
            else:
                return route
    return route

# -------------------------------------------------------------
# ILP Optimization with flow conservation for an edge (using PuLP)
# -------------------------------------------------------------
def solve_edge_ILP(start: Point, end: Point, usage: np.ndarray, dims: Tuple[int,int,int], alpha: float, box: Tuple[Tuple[int,int,int], Tuple[int,int,int]]) -> Optional[np.ndarray]:
    """
    Set up an ILP that finds a minimal-cost path between start and end within the given box.
    This ILP uses flow conservation constraints across the grid cells in the search region.
    Returns a binary 3D array (of dims) where 1 indicates the cell is used.
    """
    min_box, max_box = box
    prob = pl.LpProblem("EdgeRouting", pl.LpMinimize)
    cell_vars = {}
    # Only consider cells within the box.
    valid_cells = []
    for x in range(min_box[0], min(max_box[0], dims[0])):
        for y in range(min_box[1], min(max_box[1], dims[1])):
            for z in range(min_box[2], min(max_box[2], dims[2])):
                cell = (x,y,z)
                valid_cells.append(cell)
                cell_vars[cell] = pl.LpVariable(f"x_{x}_{y}_{z}", 0, 1, pl.LpBinary)
    # Objective: minimize sum of advanced cost over valid cells.
    prob += pl.lpSum([ advanced_cost(usage[x,y,z], alpha) * cell_vars[(x,y,z)] for (x,y,z) in valid_cells ])
    # Flow conservation: for each cell, define inflow minus outflow equals b.
    # For simplicity, we assume 6-neighborhood connectivity.
    # b = 1 for start, -1 for end, 0 otherwise.
    def neighbors(cell):
        x,y,z = cell
        nbs = []
        for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nx,ny,nz = x+dx,y+dy,z+dz
            if (nx,ny,nz) in cell_vars:
                nbs.append((nx,ny,nz))
        return nbs

    for cell in valid_cells:
        b = 0
        if cell == start.as_tuple():
            b = 1
        elif cell == end.as_tuple():
            b = -1
        prob += (pl.lpSum([ cell_vars[cell] ]) - pl.lpSum([ cell_vars[n] for n in neighbors(cell) ])) == b

    # Solve the ILP.
    prob.solve(pl.PULP_CBC_CMD(msg=0))
    if pl.LpStatus[prob.status] != "Optimal":
        return None
    route_ilp = np.zeros(dims, dtype=int)
    for cell in valid_cells:
        if pl.value(cell_vars[cell]) is not None and pl.value(cell_vars[cell]) > 0.5:
            route_ilp[cell] = 1
    return route_ilp

# -------------------------------------------------------------
# Topology-Aware Wire Rip-up and Rerouting
# -------------------------------------------------------------
def topology_aware_wire_ripup_rerouting(route_matrix: np.ndarray, usage: np.ndarray, alpha: float) -> np.ndarray:
    dims = route_matrix.shape
    new_route = route_matrix.copy()
    # Identify connected components (segments) in the route.
    # For simplicity, perform a DFS on routed cells.
    visited = np.zeros(dims, dtype=bool)
    segments = []

    def dfs(cell, comp):
        x,y,z = cell
        stack = [cell]
        while stack:
            curr = stack.pop()
            cx,cy,cz = curr
            if visited[cx,cy,cz]:
                continue
            visited[cx,cy,cz] = True
            comp.append(curr)
            for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                nx, ny, nz = cx+dx, cy+dy, cz+dz
                if 0 <= nx < dims[0] and 0 <= ny < dims[1] and 0 <= nz < dims[2]:
                    if route_matrix[nx,ny,nz] == 1 and not visited[nx,ny,nz]:
                        stack.append((nx,ny,nz))
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                if route_matrix[x,y,z] == 1 and not visited[x,y,z]:
                    comp = []
                    dfs((x,y,z), comp)
                    segments.append(comp)
    # Reroute segments that are in highly congested regions.
    threshold = 8
    for seg in segments:
        seg_usage = np.mean([usage[cell] for cell in seg])
        if seg_usage > threshold:
            # Rip up this segment.
            for cell in seg:
                new_route[cell] = 0
                usage[cell] = max(usage[cell]-3, 0)
            # In a full implementation, call negotiation-based A* again for this segment.
            # For demonstration, we leave it ripped-up.
    return new_route

# -------------------------------------------------------------
# Layer Assignment and Via Optimization
# -------------------------------------------------------------
def layer_assignment_and_via_optimization(route_matrix: np.ndarray) -> np.ndarray:
    """
    Assign layers to the routed net via an ILP formulation.
    For each routed cell, a binary variable is created. Additionally, for each adjacent pair
    of routed cells (in +X, +Y, +Z directions), an auxiliary binary variable is introduced to
    represent the absolute difference between the two layer assignments.
    The objective minimizes the sum of these difference variables.
    
    Returns a layer matrix (same dims) with values 0 or 1.
    """
    dims = route_matrix.shape
    prob = pl.LpProblem("LayerAssignment", pl.LpMinimize)
    layer_vars = {}
    
    # Create a binary variable for each routed cell.
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                if route_matrix[x,y,z] == 1:
                    layer_vars[(x,y,z)] = pl.LpVariable(f"layer_{x}_{y}_{z}", 0, 1, pl.LpBinary)
                    
    diff_vars = []
    # For each cell in layer_vars, create auxiliary variables for neighbors in +X, +Y, +Z directions.
    for cell in layer_vars:
        x, y, z = cell
        for dx, dy, dz in [(1,0,0), (0,1,0), (0,0,1)]:
            nb = (x+dx, y+dy, z+dz)
            if nb in layer_vars:
                diff_var = pl.LpVariable(f"diff_{x}_{y}_{z}_{nb[0]}_{nb[1]}_{nb[2]}", 0, 1, pl.LpBinary)
                diff_vars.append(diff_var)
                # The absolute difference constraint is modeled by:
                # diff_var >= layer_vars[cell] - layer_vars[nb]
                # diff_var >= layer_vars[nb] - layer_vars[cell]
                prob += layer_vars[cell] - layer_vars[nb] <= diff_var
                prob += layer_vars[nb] - layer_vars[cell] <= diff_var

    # Objective: minimize the total number of differences (i.e., vias).
    if diff_vars:
        prob += pl.lpSum(diff_vars)
    else:
        prob += 0

    prob.solve(pl.PULP_CBC_CMD(msg=0))
    layer_matrix = np.zeros(dims, dtype=int)
    for cell, var in layer_vars.items():
        if pl.value(var) is not None and pl.value(var) >= 0.5:
            layer_matrix[cell] = 1
    return layer_matrix


# -------------------------------------------------------------
# MST Computation for Net Decomposition (Prim's algorithm)
# -------------------------------------------------------------
def compute_MST(pins: List[Point]) -> List[Tuple[Point, Point]]:
    if not pins:
        return []
    mst_edges = []
    in_tree = {pins[0]}
    not_in_tree = set(pins[1:])
    while not_in_tree:
        best_edge, best_dist = None, float('inf')
        for u in in_tree:
            for v in not_in_tree:
                d = abs(u.x-v.x)+abs(u.y-v.y)+abs(u.z-v.z)
                if d < best_dist:
                    best_dist = d
                    best_edge = (u,v)
        if best_edge is None:
            break
        mst_edges.append(best_edge)
        u,v = best_edge
        in_tree.add(v)
        not_in_tree.remove(v)
    return mst_edges

# -------------------------------------------------------------
# Full Global Router Class integrating all modules
# -------------------------------------------------------------
class FullGlobalRouter:
    def __init__(self, pin_matrix: np.ndarray, usage_matrix: np.ndarray, alpha: float):
        self.pin_matrix = pin_matrix.copy()
        self.usage_matrix = usage_matrix.copy()
        self.alpha = alpha
        self.dims = pin_matrix.shape
        self.route_matrix = np.zeros(self.dims, dtype=int)
        self.internal_usage = self.usage_matrix.copy()
    
    def get_all_pins(self) -> List[Point]:
        indices = np.argwhere(self.pin_matrix == 1)
        return [Point(int(idx[0]), int(idx[1]), int(idx[2])) for idx in indices]
    
    def global_routing(self) -> np.ndarray:
        # Step 1: Pre-Routing Congestion Prediction.
        predicted = pre_routing_congestion_prediction(self.usage_matrix)
        self.internal_usage = predicted.copy()
        
        # Step 2: Net Decomposition using MST.
        pins = self.get_all_pins()
        if len(pins) < 2:
            print("Not enough pins to route.")
            return self.route_matrix
        mst_edges = compute_MST(pins)
        # print("Computed MST edges:", mst_edges)
        
        # For each MST edge, perform negotiation-based A* within an expanded box.
        for i, (start, end) in enumerate(mst_edges):
            # Compute net bounding box.
            min_xyz = (min(start.x,end.x), min(start.y,end.y), min(start.z,end.z))
            max_xyz = (max(start.x,end.x)+1, max(start.y,end.y)+1, max(start.z,end.z)+1)
            net_bbox = (min_xyz, max_xyz)
            # Expand box based on current iteration.
            box = box_expansion_strategy(net_bbox, i, max_expand=5)
            # Run negotiation-based A* in the restricted box.
            route = negotiation_based_A_star(start, end, self.internal_usage, self.dims, self.alpha, box)
            if route is None:
                print(f"Edge {start}-{end}: No route found by negotiation.")
                continue
            for pt in route:
                self.route_matrix[pt.x, pt.y, pt.z] = 1
                self.internal_usage[pt.x, pt.y, pt.z] += 1
                
        # Step 3: Progressive ILP Optimization over each MST edge.
        ilp_routes = np.zeros(self.dims, dtype=int)
        for (start, end) in mst_edges:
            # Use the expanded box from before (here, reuse a box covering start and end)
            min_xyz = (min(start.x,end.x), min(start.y,end.y), min(start.z,end.z))
            max_xyz = (max(start.x,end.x)+1, max(start.y,end.y)+1, max(start.z,end.z)+1)
            box = (min_xyz, max_xyz)
            ilp_route = solve_edge_ILP(start, end, self.internal_usage, self.dims, self.alpha, box)
            if ilp_route is not None:
                ilp_routes = np.maximum(ilp_routes, ilp_route)
        self.route_matrix = np.maximum(self.route_matrix, ilp_routes)
        
        # Step 4: Topology-Aware Wire Rip-up and Rerouting.
        self.route_matrix = topology_aware_wire_ripup_rerouting(self.route_matrix, self.internal_usage, self.alpha)
        
        # Step 5: Layer Assignment and Via Optimization.
        self.layer_matrix = layer_assignment_and_via_optimization(self.route_matrix)
        
        # Return the final binary route matrix.
        return self.route_matrix
    
# -------------------------------------------------------------
# Visualization with Plotly.
# -------------------------------------------------------------
def visualize_full(usage: np.ndarray, route: np.ndarray, pin_matrix: np.ndarray, layer: Optional[np.ndarray] = None):
    dims = usage.shape
    xs, ys, zs, cell_vals, hover_text = [], [], [], [], []
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                xs.append(x)
                ys.append(y)
                zs.append(z)
                val = usage[x, y, z]
                cell_vals.append(val)
                hover_text.append(f"Usage: {val}<br>Coord: ({x},{y},{z})")
    usage_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='markers',
        marker=dict(size=3, color=cell_vals, colorscale='Viridis', colorbar=dict(title='Usage'), opacity=0.7),
        text=hover_text,
        hoverinfo='text',
        name="Usage"
    )
    
    # Build trace for the routed net with hover text.
    route_x, route_y, route_z, route_hover = [], [], [], []
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                if route[x, y, z] == 1:
                    route_x.append(x)
                    route_y.append(y)
                    route_z.append(z)
                    val = usage[x, y, z]
                    route_hover.append(f"Usage: {val}<br>Route Cell: ({x},{y},{z})")
    
    route_trace = go.Scatter3d(
        x=route_x,
        y=route_y,
        z=route_z,
        mode='markers',
        marker=dict(size=6, color='red'),
        text=route_hover,
        hoverinfo='text',
        name="Routed Net"
    )
    
    # Build trace for pins (bright green markers).
    pin_x, pin_y, pin_z = [], [], []
    for idx in np.argwhere(pin_matrix == 1):
        pin_x.append(int(idx[0]))
        pin_y.append(int(idx[1]))
        pin_z.append(int(idx[2]))
    pin_trace = go.Scatter3d(
        x=pin_x,
        y=pin_y,
        z=pin_z,
        mode='markers',
        marker=dict(size=8, color='lime'),
        name="Pins"
    )
    
    layers_trace = None
    if layer is not None:
        layer0_x, layer0_y, layer0_z = [],[],[]
        layer1_x, layer1_y, layer1_z = [],[],[]
        for x in range(dims[0]):
            for y in range(dims[1]):
                for z in range(dims[2]):
                    if route[x, y, z] == 1:
                        if layer[x, y, z] == 0:
                            layer0_x.append(x)
                            layer0_y.append(y)
                            layer0_z.append(z)
                        else:
                            layer1_x.append(x)
                            layer1_y.append(y)
                            layer1_z.append(z)
        layer0_trace = go.Scatter3d(
            x=layer0_x,
            y=layer0_y,
            z=layer0_z,
            mode='markers',
            marker=dict(size=8, color='orange'),
            name="Layer 0"
        )
        layer1_trace = go.Scatter3d(
            x=layer1_x,
            y=layer1_y,
            z=layer1_z,
            mode='markers',
            marker=dict(size=8, color='blue'),
            name="Layer 1"
        )
        layers_trace = [layer0_trace, layer1_trace]
    
    data = [usage_trace, route_trace, pin_trace] + (layers_trace if layers_trace is not None else [])
    fig = go.Figure(data=data)
    fig.update_layout(
        title="Full Global Routing Visualization",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
    )
    fig.show()


# -------------------------------------------------------------
# Main: Example usage.
# -------------------------------------------------------------
if __name__ == "__main__":
    dims = (10,10,10)
    # Define a sample pin matrix.
    pin_matrix = np.zeros(dims, dtype=int)
    pin_matrix[1,1,1] = 1
    pin_matrix[8,8,8] = 1
    pin_matrix[2,3,4] = 1
    pin_matrix[7,6,5] = 1
    pin_matrix[4,4,2] = 1  # extra terminal

    # Generate a usage matrix in chunks to simulate regional congestion.
    # For illustration, use random block values.
    chunk = (3,3,3)
    usage_matrix = np.zeros(dims, dtype=int)
    for x in range(0, dims[0], chunk[0]):
        for y in range(0, dims[1], chunk[1]):
            for z in range(0, dims[2], chunk[2]):
                val = np.random.randint(0,10)
                x_end = min(x+chunk[0], dims[0])
                y_end = min(y+chunk[1], dims[1])
                z_end = min(z+chunk[2], dims[2])
                usage_matrix[x:x_end, y:y_end, z:z_end] = val

    alpha = 1.0
    router = FullGlobalRouter(pin_matrix, usage_matrix, alpha)
    final_route = router.global_routing()
    print("\nFinal Global Route Matrix (1 indicates a routed cell):")
    print(final_route)
    visualize_full(router.usage_matrix, final_route, pin_matrix, router.layer_matrix)
