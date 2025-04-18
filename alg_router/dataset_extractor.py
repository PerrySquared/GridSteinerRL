import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
np.set_printoptions(threshold=sys.maxsize)
random.seed(11)

MATRIX_SIZE = 32

def display_heatmap(matrix):
    # print(type(matrix))
    plt.imshow(matrix, cmap='hot')
    plt.show()
    plt.close()

def get_coordinates(matrix):
    coordinates = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 1:
                coordinates.append(np.array([i, j]))
    return coordinates


def get_coords_dataset(start, target_amount, f):
    # Open the HDF5 file
        
        
        group_keys = list(f.keys())
        
        for i in range(start, len(group_keys)):
            # Access the group

            key = group_keys[i]
            group = f[key]
            # Extract the matrix from the random dataset
            matrix = np.zeros([MATRIX_SIZE, MATRIX_SIZE])
            matrix = group[key][:]  # Access the "data" dataset within the group
            
            # plt.imshow(matrix)
            # plt.show()
            
            net_name = group.attrs.get("net_name", "Unknown")
            insertion_coords = group.attrs.get("insertion_coords", (0, 0))
            origin_shape = group.attrs.get("origin_shape", (0, 0))

            # Check if the matrix has MATRIX_SIZE rows or less and MATRIX_SIZE columns or less
            target_count_on_matrix = np.count_nonzero(matrix == 1)
            if matrix.shape[0] <= MATRIX_SIZE and matrix.shape[1] <= MATRIX_SIZE and target_count_on_matrix == target_amount:
                # Pad the matrix if necessary
                pad_width = ((0, max(0, MATRIX_SIZE - matrix.shape[0])), (0, max(0, MATRIX_SIZE - matrix.shape[1])))
                padded_matrix = np.pad(matrix, pad_width, mode='constant', constant_values=0)
                # plt.imshow(padded_matrix)
                # plt.show() 
                
                # print(get_coordinates(padded_matrix), net_name, insertion_coords, origin_shape, i)
                return get_coordinates(padded_matrix), net_name, insertion_coords, origin_shape, i
            # else:
            #     pass
            #     # return get_coords_dataset()


def get_coordinates_3d(matrix):
    coordinates = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                if matrix[i, j, k] == 1:
                    coordinates.append(np.array([i, j, k]))
    return coordinates


def get_coords_dataset_3d(start, target_amount, f, limited_size=True):

    group_keys = list(f.keys())
    
    for i in range(start, len(group_keys)):

        key = group_keys[i]
        group = f[key]

        LAYERS = 6
        matrix = np.zeros([MATRIX_SIZE, MATRIX_SIZE, LAYERS]) 
        matrix = group[key][:]  
        
        # print(matrix.shape)
        net_name = group.attrs.get("net_name", "Unknown")
        insertion_coords = group.attrs.get("insertion_coords", (0, 0, 0))
        origin_shape = group.attrs.get("origin_shape", (0, 0, 0))

        target_count_on_matrix = np.count_nonzero(matrix == 1)
        
        if limited_size:
            if (matrix.shape[0] < MATRIX_SIZE and 
                matrix.shape[1] < MATRIX_SIZE and 
                target_count_on_matrix == target_amount):
                
                pad_width = ((0, max(0, MATRIX_SIZE - matrix.shape[0])),
                            (0, max(0, MATRIX_SIZE - matrix.shape[1])),
                            (0, 0))
                
                padded_matrix = np.pad(matrix, pad_width, mode='constant', constant_values=0)
                
                insertion_coords = np.append(insertion_coords, 0)
                
                return get_coordinates_3d(padded_matrix), net_name, insertion_coords, origin_shape, matrix.shape, i
        else:
            if (matrix.shape[0] >= MATRIX_SIZE and 
                matrix.shape[1] >= MATRIX_SIZE):
                
                insertion_coords = np.append(insertion_coords, 0)
                return get_coordinates_3d(matrix), net_name, insertion_coords, origin_shape, matrix.shape, i
            
    return False, False, False, False, False        
    
# import nexusformat.nexus as nx
# f = nx.nxload('./dataset.h5')
# print(f.tree)

# get_coords_dataset(0)

# if __name__ == "__main__":
#     cur_max = 0
    
#     for i in range(10000):
#         # print(i)
#         out = get_coords_dataset()
#         if len(out) > cur_max:
#             cur_max = len(out)
#             print(cur_max)
            
#         matrix = np.zeros((32,32), dtype=int)
#         for _target_location in out:
#             # place_support_nodes(out)
#             matrix[_target_location[0], _target_location[1]] = 1
            
        # display_heatmap(matrix)

