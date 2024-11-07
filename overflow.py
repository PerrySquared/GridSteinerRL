import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

matrix = [
    [  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16],
    [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
    [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64],
    [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
    [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96],
    [97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
    [113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128],
    [129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144],
    [145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160],
    [161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176],
    [177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192],
    [193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208],
    [209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224],
    [225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240],
    [241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256]
]

np_matrix = np.asarray(matrix, dtype=np.float32)
print(np_matrix)
"""
# general overflow matrix - gcell overflow matrix for a whole schema
# reference overflow matrix - gcell overflow matrix for a small 
# standartized area that contains the picked net and is cut out from general overflow matrix 
# local overflow matrix - gcell overflow matrix for paths created by algorithm (or RL NN) - 
# basically the same thing as path matrix because initially it's empty and is filled with a 
# constant value in places where created path is located
"""

OVERFLOW_VALUE = 1

class OverflowWithOverlap():
    def __init__(self, general_overflow_matrix, rows, columns):
        """
        local overflow is the same thing as path due to it being an empty matrix 
        that is getting filled in the same cells as path with some identical 
        values that represent overflow
        """
        self.local_overflow_matrix = np.zeros([rows, columns]) 
        self.overflow_reference_matrix = self.get_overflow_reference_matrix()

    def get_overflow_reference_matrix(self):

        # some code to get a cutout of a standertized shape from general overflow matrix in required place
        # rewrite dataser_extractor in a way that gets the origin coordinates of the cutout to later place 
        # the local overflow matrix into the global matrix
        overflow_reference_matrix = True

        return overflow_reference_matrix

    def get_manhattan_path(self, overflow_reference_matrix, start, end):
        x1, y1 = start
        x2, y2 = end
        
        path_sum = 0    

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
                if self.local_overflow_matrix[y1][x] == 0:
                    path_sum += overflow_reference_matrix[y1][x]
                self.local_overflow_matrix[y1][x] = OVERFLOW_VALUE 
            # Sum the values moving down
            for y in range(y1 + 1, y2 + 1):
                if self.local_overflow_matrix[y][x2] == 0:
                    path_sum += overflow_reference_matrix[y][x2]
                self.local_overflow_matrix[y][x2] = OVERFLOW_VALUE

        if y1 > y2:
            # Move down first
            # Sum the values moving down
            for y in range(y2, y1):
                if self.local_overflow_matrix[y][x2] == 0:
                    path_sum += overflow_reference_matrix[y][x2]
                self.local_overflow_matrix[y][x2] = OVERFLOW_VALUE
            # Sum the values moving right
            for x in range(min(x1, x2), max(x1 + 1, x2 + 1)):
                if self.local_overflow_matrix[y1][x] == 0:
                    path_sum += overflow_reference_matrix[y1][x]
                self.local_overflow_matrix[y1][x] = OVERFLOW_VALUE


        return path_sum, self.local_overflow_matrix

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

c = OverflowWithOverlap(True, 16,16)

print(c.get_manhattan_path(matrix, (3,5), (8,9)))
print(c.get_manhattan_path(matrix, (8,9), (10,3)))
print(c.get_manhattan_path(matrix, (10,3), (6,2)))
print(c.get_manhattan_path(matrix, (6,2), (1,7)))
print(c.get_manhattan_path(matrix, (1,7), (6,0)))

# print(c.get_both_get_manhattan_paths(matrix, (8,9), (13,5)))
# print(get_both_get_manhattan_paths(matrix, (8,9), (3,15)))
# print(get_both_get_manhattan_paths(matrix, (8,9), (13,15)))





