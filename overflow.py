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

<<<<<<< HEAD

=======
>>>>>>> 662b995cdf15b553601f1305b7e0e988b7eec8b8
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
        normalized_path_sum = 0 

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
                    normalized_path_sum += (self.overflow_reference_matrix[x][y1] - np.min(self.overflow_reference_matrix))/denominator
                    
                self.local_overflow_matrix[x][y1] = OVERFLOW_VALUE 
            # Sum the values moving down
            for y in range(y1 + 1, y2 + 1):
                if self.local_overflow_matrix[x2][y] == 0:
                    path_sum += self.overflow_reference_matrix[x2][y]
                    normalized_path_sum += (self.overflow_reference_matrix[x2][y] - np.min(self.overflow_reference_matrix))/denominator
                    
                self.local_overflow_matrix[x2][y] = OVERFLOW_VALUE

        if y1 > y2:
            # Move down first
            # Sum the values moving down
            for y in range(y2, y1):
                if self.local_overflow_matrix[x2][y] == 0:
                    path_sum += self.overflow_reference_matrix[x2][y]
                    normalized_path_sum += (self.overflow_reference_matrix[x2][y] - np.min(self.overflow_reference_matrix))/denominator
                    
                self.local_overflow_matrix[x2][y] = OVERFLOW_VALUE
            # Sum the values moving right
            for x in range(min(x1, x2), max(x1 + 1, x2 + 1)):
                if self.local_overflow_matrix[x][y1] == 0:
                    path_sum += self.overflow_reference_matrix[x][y1]
                    normalized_path_sum += (self.overflow_reference_matrix[x][y1] - np.min(self.overflow_reference_matrix))/denominator
                    
                self.local_overflow_matrix[x][y1] = OVERFLOW_VALUE

        return path_sum, normalized_path_sum, path_length, self.local_overflow_matrix

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

    def add_path_section_to_path_only_matrix(self, cell):
        x, y = cell
        self.path_only_matrix[x][y] = 1
        
    def display_matrices(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

        # Display each matrix in its subplot
        axes[0].imshow(self.general_overflow_matrix, aspect='auto')  # 'aspect' can be adjusted
        axes[0].set_title('General Overflow Matrix')

        axes[1].imshow(self.overflow_reference_matrix, aspect='auto')
        axes[1].set_title('Overflow Reference Matrix')

        axes[2].imshow(self.local_overflow_matrix, aspect='auto')
        axes[2].set_title('Local Overflow Matrix')


        # Show the plot with all three matrices
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

# general_overflow_matrix = np.zeros([16, 16]) 
# local_overflow_matrix = np.zeros([16, 16]) 

# c = OverflowWithOverlap(general_overflow_matrix, local_overflow_matrix, 16, 16)

# print(c.get_manhattan_path((3,5), (8,9)))
# print(c.get_manhattan_path((8,9), (10,3)))
# print(c.get_manhattan_path((10,3), (6,2)))
# print(c.get_manhattan_path((6,2), (1,7)))
# print(c.get_manhattan_path((1,7), (6,0)))






