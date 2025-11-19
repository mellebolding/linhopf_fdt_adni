import numpy as np
from scipy.io import loadmat

# --- Configuration ---
# IMPORTANT: Replace this path with the actual path to your .mat file.
# We will use one of the filenames you uploaded as an example.
MAT_values = 'data/ADNI-B_DATA/N238rev/tseries/sch400/tseries_ADNI3_AD_MPRAGE_batches123_sch400_matching_QC_COMBINED.mat'
MAT_A = 'data/ADNI-B_DATA/N238rev/tseries/sch400/combined_PTIDS_ADNI3_AD_MPRAGE.mat'


def inspect_mat_file(file_path):
    """
    Loads a .mat file, prints its dictionary keys, and provides a shape and type 
    summary for the top-level variables.
    
    :param file_path: The path to the .mat file.
    """
    print(f"--- Inspecting file: {file_path} ---")

    try:
        # Load the .mat file content into a Python dictionary
        mat_contents = loadmat(file_path)
    except FileNotFoundError:
        print(f"ERROR: File not found at path: {file_path}")
        print("Please ensure the file path is correct.")
        return
    except Exception as e:
        print(f"ERROR: Could not load .mat file. Ensure it is a valid MATLAB file (v5 or higher). Details: {e}")
        return

    # 1. Print all top-level keys
    print("\n[1] Top-Level Dictionary Keys (Variables):")
    # Exclude standard internal MATLAB metadata keys
    keys = [key for key in mat_contents.keys() if not key.startswith('__')]
    print(keys)
    
    print("\n[2] Variable Type and Shape Summary:")
    
    # 2. Iterate through keys and print structural info
    for key in keys:
        variable = mat_contents[key]
        
        # Check if the variable is a NumPy array (most common case for data)
        if isinstance(variable, np.ndarray):
            print(f"- Key: '{key}'")
            print(f"  Type: {variable.dtype}")
            print(f"  Shape: {variable.shape}")
            
            # If the array is very large, print only the first few elements
            if variable.size > 0:
                print(f"  Example data (first few values): {list(variable.flat[:min(5, variable.size)])}")
            
            # Access and print the actual array elements if they are nested
            if variable.dtype == np.dtype('object'):
                print("  Accessing nested array elements:")
            for i, elem in enumerate(variable.flat[:min(5, variable.size)]):
                if isinstance(elem, np.ndarray):
                    print(f"    Element {i}: {list(elem)}")
                else:
                    print(f"    Element {i}: {elem}")
        else:
            # Handle non-array types (should be rare for data files)
            print(f"- Key: '{key}'")
            print(f"  Type: {type(variable)}")
            
    print("\n--- Inspection Complete ---")

# Execute the inspection function
if __name__ == "__main__":
    inspect_mat_file(MAT_values)