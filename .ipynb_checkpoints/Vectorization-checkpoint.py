import numpy as np

def covariance(X, Y):
    """
    Calculate the covariance between two arrays X and Y, ignoring NaN values, in a vectorized manner.
    Assumes X and Y have the shape (lat, lon, time).
    """
    # Calculate means along the time axis while ignoring NaNs
    mean_X = np.nanmean(X, axis=2, keepdims=True)
    mean_Y = np.nanmean(Y, axis=2, keepdims=True)

    # Subtract the mean from each variable
    X_centered = X - mean_X
    Y_centered = Y - mean_Y

    # Count the number of valid (non-NaN) observations along the time axis
    n_valid = np.sum(~np.isnan(X) & ~np.isnan(Y), axis=2)

    # Calculate the covariance term while ignoring NaNs and applying Bessel's correction
    covariance = np.nansum(X_centered * Y_centered, axis=2) / (n_valid - 1)

    return covariance

def correlation(X, Y):
    """
    Calculate the correlation coefficient between two arrays X and Y, ignoring NaN values, in a vectorized manner.
    Assumes X and Y have the shape (lat, lon, time).
    """
    # Calculate the variances using the nan_covariance function
    covXX = covariance(X, X)
    covYY = covariance(Y, Y)

    # Calculate the covariance between X and Y
    covXY = covariance(X, Y)

    # Calculate the correlation coefficient
    correlation = covXY / np.sqrt(covXX * covYY)

    return correlation

def cov_corr(X, Y, calc_cov=True, calc_corr=True):
    """
    Calculate the covariance, correlation, or both between two arrays X and Y, ignoring NaN values, in a vectorized manner.
    Assumes X and Y have the shape (lat, lon, time).
    
    Parameters:
    - X, Y: Input arrays with shape (lat, lon, time)
    - calc_cov: Boolean indicating whether to calculate covariance
    - calc_corr: Boolean indicating whether to calculate correlation
    
    Returns:
    - covXY: Covariance matrix (if calc_cov is True)
    - corrXY: Correlation matrix (if calc_corr is True)
    """
    # Initialize return values
    covXY = None
    corrXY = None

    # Calculate means along the time axis while ignoring NaNs
    mean_X = np.nanmean(X, axis=2, keepdims=True)
    mean_Y = np.nanmean(Y, axis=2, keepdims=True)

    # Subtract the mean from each variable
    X_centered = X - mean_X
    Y_centered = Y - mean_Y

    # Count the number of valid (non-NaN) observations along the time axis
    n_valid = np.sum(~np.isnan(X) & ~np.isnan(Y), axis=2)

    # Calculate covariance if requested
    if calc_cov or calc_corr:  # Calculate covariance if needed for either output
        covXY = np.nansum(X_centered * Y_centered, axis=2) / (n_valid - 1)

    # Calculate correlation if requested
    if calc_corr:
        # Calculate the variances
        covXX = np.nansum(X_centered * X_centered, axis=2) / (n_valid - 1)
        covYY = np.nansum(Y_centered * Y_centered, axis=2) / (n_valid - 1)
        
        # Calculate the correlation coefficient
        corrXY = covXY / np.sqrt(covXX * covYY)

    # Return the requested values
    if calc_cov and calc_corr:
        return covXY, corrXY
    elif calc_cov:
        return covXY
    elif calc_corr:
        return corrXY

def cov_corr_three_old(X, Y, Z, calc_cov=True, calc_corr=True):
    """
    Calculate the covariance, correlation, or both for three arrays X, Y, and Z, ignoring NaN values, in a vectorized manner.
    Assumes X, Y, and Z have the shape (lat, lon, time).
    
    Parameters:
    - X, Y, Z: Input arrays with shape (lat, lon, time)
    - calc_cov: Boolean indicating whether to calculate covariance
    - calc_corr: Boolean indicating whether to calculate correlation
    
    Returns:
    - results: A dictionary containing covariances and correlations (depending on the flags)
    """
    results = {}

    # Calculate means along the time axis while ignoring NaNs
    mean_X = np.nanmean(X, axis=2, keepdims=True)
    mean_Y = np.nanmean(Y, axis=2, keepdims=True)
    mean_Z = np.nanmean(Z, axis=2, keepdims=True)

    # Subtract the mean from each variable
    X_centered = X - mean_X
    Y_centered = Y - mean_Y
    Z_centered = Z - mean_Z

    # Count the number of valid (non-NaN) observations along the time axis
    n_valid_XY = np.sum(~np.isnan(X) & ~np.isnan(Y), axis=2)
    n_valid_YZ = np.sum(~np.isnan(Y) & ~np.isnan(Z), axis=2)
    n_valid_XZ = np.sum(~np.isnan(X) & ~np.isnan(Z), axis=2)

    # Calculate covariances between different variables
    if calc_cov or calc_corr:
        results['covXY'] = np.nansum(X_centered * Y_centered, axis=2) / (n_valid_XY - 1)
        results['covYZ'] = np.nansum(Y_centered * Z_centered, axis=2) / (n_valid_YZ - 1)
        results['covXZ'] = np.nansum(X_centered * Z_centered, axis=2) / (n_valid_XZ - 1)

    # Calculate correlations
    if calc_corr:
        # Calculate variances (covariance of each variable with itself)
        covXX = np.nansum(X_centered * X_centered, axis=2) / (n_valid_XY - 1)
        covYY = np.nansum(Y_centered * Y_centered, axis=2) / (n_valid_XY - 1)
        covZZ = np.nansum(Z_centered * Z_centered, axis=2) / (n_valid_YZ - 1)
        # Store variances in the results dictionary
        results['covXX'] = covXX
        results['covYY'] = covYY
        results['covZZ'] = covZZ
    
        results['corrXY'] = results['covXY'] / np.sqrt(covXX * covYY)
        results['corrYZ'] = results['covYZ'] / np.sqrt(covYY * covZZ)
        results['corrXZ'] = results['covXZ'] / np.sqrt(covXX * covZZ)

    return results

def cov_corr_three(X, Y, Z, calc_cov=True, calc_corr=True):
    results = {}
    X_2d = X.reshape(-1, X.shape[2])
    Y_2d = Y.reshape(-1, Y.shape[2])
    Z_2d = Z.reshape(-1, Z.shape[2])

    valid_mask = ~np.isnan(X_2d).all(axis=1) & ~np.isnan(Y_2d).all(axis=1) & ~np.isnan(Z_2d).all(axis=1)
    X_valid = X_2d[valid_mask]
    Y_valid = Y_2d[valid_mask]
    Z_valid = Z_2d[valid_mask]

    mean_X = np.nanmean(X_valid, axis=1, keepdims=True)
    mean_Y = np.nanmean(Y_valid, axis=1, keepdims=True)
    mean_Z = np.nanmean(Z_valid, axis=1, keepdims=True)

    X_centered = X_valid - mean_X
    Y_centered = Y_valid - mean_Y
    Z_centered = Z_valid - mean_Z

    n_valid_XY = np.sum(~np.isnan(X_valid) & ~np.isnan(Y_valid), axis=1)
    n_valid_YZ = np.sum(~np.isnan(Y_valid) & ~np.isnan(Z_valid), axis=1)
    n_valid_XZ = np.sum(~np.isnan(X_valid) & ~np.isnan(Z_valid), axis=1)

    original_shape = (X.shape[0], X.shape[1])

    if calc_cov:
        covXY = np.nansum(X_centered * Y_centered, axis=1) / (n_valid_XY - 1)
        covYZ = np.nansum(Y_centered * Z_centered, axis=1) / (n_valid_YZ - 1)
        covXZ = np.nansum(X_centered * Z_centered, axis=1) / (n_valid_XZ - 1)
        
        results['covXY'] = np.full(original_shape, np.nan)
        results['covYZ'] = np.full(original_shape, np.nan)
        results['covXZ'] = np.full(original_shape, np.nan)
        
        results['covXY'][valid_mask.reshape(original_shape)] = covXY
        results['covYZ'][valid_mask.reshape(original_shape)] = covYZ
        results['covXZ'][valid_mask.reshape(original_shape)] = covXZ

    if calc_corr:
        covXX = np.nansum(X_centered * X_centered, axis=1) / (n_valid_XY - 1)
        covYY = np.nansum(Y_centered * Y_centered, axis=1) / (n_valid_XY - 1)
        covZZ = np.nansum(Z_centered * Z_centered, axis=1) / (n_valid_YZ - 1)
    
        corrXY = covXY / np.sqrt(covXX * covYY)
        corrYZ = covYZ / np.sqrt(covYY * covZZ)
        corrXZ = covXZ / np.sqrt(covXX * covZZ)
        
        results['covXX'] = np.full(original_shape, np.nan)
        results['covYY'] = np.full(original_shape, np.nan)
        results['covZZ'] = np.full(original_shape, np.nan)
        results['corrXY'] = np.full(original_shape, np.nan)
        results['corrYZ'] = np.full(original_shape, np.nan)
        results['corrXZ'] = np.full(original_shape, np.nan)
        
        results['covXX'][valid_mask.reshape(original_shape)] = covXX
        results['covYY'][valid_mask.reshape(original_shape)] = covYY
        results['covZZ'][valid_mask.reshape(original_shape)] = covZZ
        results['corrXY'][valid_mask.reshape(original_shape)] = corrXY
        results['corrYZ'][valid_mask.reshape(original_shape)] = corrYZ
        results['corrXZ'][valid_mask.reshape(original_shape)] = corrXZ

    return results