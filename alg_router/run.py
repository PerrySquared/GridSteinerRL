import numpy as np
from boxrouter_gr import FullGlobalRouter
from dataset_extractor import get_coords_dataset_3d
import copy
import h5py
import time


TEMP_GENERAL_OVERFLOW = np.zeros((1000, 1000, 6), dtype=np.float64)
TARGETS_TOTAL = 2
TERMINAL_CELL = 1

found_instance_index = 0
f = h5py.File('./dataset_3d.h5', 'r')

print("file read")

start = time.time()

for i in range(500):
    _target_locations_copy = np.full((TARGETS_TOTAL, 3), -1)  # Updated to 3D coordinates
    # Get 3D coordinates from the dataset
    temp_target_locations_copy, net_name, insertion_coords, origin_shape, found_instance_index = get_coords_dataset_3d(found_instance_index + 1, TARGETS_TOTAL, f)        

    TARGETS_TOTAL = len(temp_target_locations_copy)
    _target_locations_copy[:TARGETS_TOTAL] = temp_target_locations_copy
        


    x_start, y_start, z_start = insertion_coords
    rows, columns, layers = 32, 32, 6

    temp_matrix = TEMP_GENERAL_OVERFLOW[
        x_start:x_start + rows, 
        y_start:y_start + columns,
        z_start:z_start + layers 
    ]

    current_rows, current_cols, current_layers = temp_matrix.shape

    # Copy the existing matrix into the new matrix to ensure defined dimensions
    overflow_reference_matrix = np.zeros((rows, columns, layers))
    overflow_reference_matrix[:current_rows, :current_cols, :current_layers] = temp_matrix

    _target_locations = copy.deepcopy(_target_locations_copy) # copy to have an independent instance to interact with


    local_overflow_matrix = np.zeros((rows, columns, layers))

    for _target_location in _target_locations:
        if _target_location[0] != -1 and _target_location[1] != -1 and _target_location[2] != -1:
            # print(_target_location[0], _target_location[1], _target_location[2])
            local_overflow_matrix[_target_location[0], _target_location[1], _target_location[2]] = TERMINAL_CELL

        
    alpha = 1.0

    router = FullGlobalRouter(local_overflow_matrix, overflow_reference_matrix, alpha)
    final_route = router.global_routing()


    upto_row = min(insertion_coords[0] + rows, origin_shape[0])
    upto_column = min(insertion_coords[1] + columns, origin_shape[1])
    upto_z = min(insertion_coords[2] + layers, origin_shape[2])

    limit_row = origin_shape[0] - insertion_coords[0]
    limit_column = origin_shape[1] - insertion_coords[1]
    limit_z = origin_shape[2] - insertion_coords[2]

    TEMP_GENERAL_OVERFLOW[
        insertion_coords[0]:upto_row, 
        insertion_coords[1]:upto_column,
        insertion_coords[2]:upto_z
    ] += final_route[0:limit_row, 0:limit_column, 0:limit_z]

end = time.time()

print(end-start)