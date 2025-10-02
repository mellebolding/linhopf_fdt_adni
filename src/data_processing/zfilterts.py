import numpy as np
from scipy.signal import detrend as scipy_detrend
from scipy.signal import butter, filtfilt


def filter_time_series(data, bfilt, afilt, detrend=True):
    """
    Optionally detrend and filter time series using filtfilt.

    Parameters:
    - data: np.ndarray of shape [NSUB, NPARCELS, NTIMES] or [NPARCELS, NTIMES]
    - bfilt, afilt: filter coefficients for filtfilt
    - detrend: bool, whether to detrend each time series before filtering

    Returns:
    - filtered_data: np.ndarray of the same shape, filtered time series
    """
    data = np.asarray(data)
    if data.ndim == 2:
        # Single subject case: [NPARCELS, NTIMES]
        NPARCELS, NTIMES = data.shape
        filtered_data = np.zeros_like(data)
        for parcel in range(NPARCELS):
            ts = data[parcel, :]
            if detrend:
                ts = scipy_detrend(ts, type='linear')
            ts = ts - np.mean(ts)
            filtered_data[parcel, :] = filtfilt(bfilt, afilt, ts)
    elif data.ndim == 3:
        # Group case: [NSUB, NPARCELS, NTIMES]
        NSUB, NPARCELS, NTIMES = data.shape
        filtered_data = np.zeros_like(data)
        for sub in range(NSUB):
            for parcel in range(NPARCELS):
                ts = data[sub, parcel, :]
                if detrend:
                    ts = scipy_detrend(ts, type='linear')
                ts = ts - np.mean(ts)
                filtered_data[sub, parcel, :] = filtfilt(bfilt, afilt, ts)
    else:
        raise ValueError("Input data must be 2D or 3D.")
    return filtered_data

def zscore_time_series(data, mode='parcel', detrend=False):
    """
    Optionally detrend and z-score the time series either parcel-wise or globally.

    Parameters:
    - data: numpy array of shape [NSUB, NPARCELS, NTIMES] or [NPARCELS, NTIMES]
    - mode: str, either 'parcel' (default) or 'global'
        - 'parcel': z-score each parcel individually across time
        - 'global': z-score the entire time series (per subject if 3D)
        - 'none': leave unchanged
    - detrend: bool, whether to remove linear trend along the time axis

    Returns:
    - processed_data: numpy array of the same shape, detrended and/or z-scored
    """
    if mode not in ['parcel', 'global', 'none']:
        raise ValueError("mode must be 'parcel', 'global' or 'none'")

    data = np.asarray(data)

    # Apply detrending if requested
    if detrend:
        if data.ndim == 2:
            # [NPARCELS, NTIMES]
            data = scipy_detrend(data, axis=1, type='linear')
        elif data.ndim == 3:
            # [NSUB, NPARCELS, NTIMES]
            # Detrend along the time axis (axis=2) for each parcel of each subject
            data = scipy_detrend(data, axis=2, type='linear')
        else:
            raise ValueError("Input data must be 2D or 3D for detrending.")

    # Apply z-scoring
    if mode == 'none':
        return data
    if data.ndim == 2:
        if mode == 'parcel':
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
        elif mode == 'global':
            mean = np.mean(data, keepdims=True)
            std = np.std(data, keepdims=True)
    elif data.ndim == 3:
        if mode == 'parcel':
            mean = np.mean(data, axis=2, keepdims=True)
            std = np.std(data, axis=2, keepdims=True)
        elif mode == 'global':
            mean = np.mean(data, axis=(1, 2), keepdims=True)
            std = np.std(data, axis=(1, 2), keepdims=True)
    else:
        raise ValueError("Input data must be 2D or 3D.")

    std[std == 0] = 1.0  # avoid division by zero
    return (data - mean) / std