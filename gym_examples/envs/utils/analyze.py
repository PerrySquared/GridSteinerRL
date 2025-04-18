import numpy as np
import h5py
from collections import defaultdict

def analyze_dataset_statistics(file_path, start=0, end=None):
    """
    Analyze the 3D dataset to gather statistics on:
    1. How many arrays have exactly N number of 1s in them
    2. How frequently each layer is used by 1s
    
    Args:
        file_path (str): Path to the dataset file
        start (int): Starting index for processing
        end (int, optional): Ending index for processing (defaults to all)
    
    Returns:
        tuple: (ones_count_dict, layer_usage_dict) containing the statistics
    """
    # Open the dataset file
    with h5py.File(file_path, 'r') as f:
        group_keys = list(f.keys())
        
        if end is None or end > len(group_keys):
            end = len(group_keys)
        
        # Dictionaries to store statistics
        ones_count_dict = defaultdict(int)  # Key: number of 1s, Value: count of matrices
        layer_usage_dict = defaultdict(int)  # Key: layer index, Value: count of matrices
        
        # Process each item in the dataset
        for i in range(start, end):
            key = group_keys[i]
            try:
                group = f[key]
                matrix = group[key][:]
                
                # Count the number of 1s in the entire matrix
                total_ones = np.sum(matrix == 1)
                ones_count_dict[total_ones] += 1
                
                # Count which layers have at least one 1
                for layer_idx in range(matrix.shape[2]):
                    if np.any(matrix[:, :, layer_idx] == 1):
                        layer_usage_dict[layer_idx] += 1
                        
            except Exception as e:
                print(f"Error processing key {key}: {e}")
    
    return ones_count_dict, layer_usage_dict

def main():
    """
    Main function to run the analysis.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path> [start_index] [end_index]")
        sys.exit(1)
        
    file_path = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    # Run the analysis
    ones_count_dict, layer_usage_dict = analyze_dataset_statistics(file_path, start, end)
    
    # Print results in the requested format
    ones_frequency = sorted(ones_count_dict.items())
    layer_frequency = sorted(layer_usage_dict.items())
    
    print("1s frequencies in arrays:", ones_frequency)
    print("Layer usage frequency:", layer_frequency)

if __name__ == "__main__":
    main()