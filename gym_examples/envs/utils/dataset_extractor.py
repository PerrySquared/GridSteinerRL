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


def get_coords_dataset():
    # Open the HDF5 file
    with h5py.File('D:/PROJECTS/Python_projects/gym-examples-main/gym_examples/envs/utils/dataset_no_duplicates.h5', 'r') as f:
        # Get the keys of datasets in the file
        dataset_keys = list(f.keys())
        
        # Choose a random dataset
        random_dataset_key = random.choice(dataset_keys)
        
        # Extract the matrix from the random dataset
        matrix = np.zeros([MATRIX_SIZE, MATRIX_SIZE])
        matrix = f[random_dataset_key][()]
        
        # Check if the matrix has MATRIX_SIZE rows or less and MATRIX_SIZE columns or less
        if matrix.shape[0] <= MATRIX_SIZE and matrix.shape[1] <= MATRIX_SIZE and np.count_nonzero(matrix == 1) == 5:
            # Pad the matrix if necessary
            pad_width = ((0, max(0, MATRIX_SIZE - matrix.shape[0])), (0, max(0, MATRIX_SIZE - matrix.shape[1])))
            padded_matrix = np.pad(matrix, pad_width, mode='constant', constant_values=0)

            # print(padded_matrix)
            return get_coordinates(padded_matrix)
        else:
            return get_coords_dataset()
    
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

