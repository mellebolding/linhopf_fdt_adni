import os
import numpy as np

def append_record_to_npz(folder, filename, **record):
    """
    Appends a record (dict) to a 'records' array in a .npz file located in `folder`.
    Creates the folder and file if they don't exist.

    Parameters
    ----------
    folder : str
        Path to the subfolder where the file will be saved.
    filename : str
        Name of the .npz file (e.g., 'Ceff_sigma_results.npz').
    record : dict
        Arbitrary key-value pairs to store (arrays, strings, numbers, etc.).
    """
    os.makedirs(folder, exist_ok=True)  # ensure subfolder exists
    filepath = os.path.join(folder, filename)

    if os.path.exists(filepath):
        existing_data = dict(np.load(filepath, allow_pickle=True))
        records = list(existing_data.get("records", []))
    else:
        records = []

    records.append(record)
    np.savez(filepath, records=np.array(records, dtype=object))

def clear_npz_file(folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    # Save an empty records array, overwriting any existing file
    np.savez(filepath, records=np.array([], dtype=object))