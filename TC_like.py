import numpy as np
import HydroAI.Vectorization as hVec

def TCA(D1, D2, D3, nod_th=20, corr_th=0.1, REF=None):
    
    avail_D1 = ~np.isnan(D1)
    avail_D2 = ~np.isnan(D2)
    avail_D3 = ~np.isnan(D3)
    avail = avail_D1 & avail_D2 & avail_D3
    avail_D1 = None
    avail_D2 = None
    avail_D3 = None

    if REF is not None:
        avail_REF = ~np.isnan(REF)
        avail = avail & avail_REF
        avail_REF = None

    avail = np.sum(avail, axis=2) > nod_th
    flag_check = np.full((D1.shape[0], D1.shape[1], 3), np.nan)
    index1, index2 = np.where(avail)
    VAR_err_x_2d = np.full((D1.shape[0], D1.shape[1]), np.nan)
    VAR_err_y_2d = np.copy(VAR_err_x_2d)
    VAR_err_z_2d = np.copy(VAR_err_x_2d)
    SNR_x_2d = np.copy(VAR_err_x_2d)
    SNR_y_2d = np.copy(SNR_x_2d)
    SNR_z_2d = np.copy(SNR_x_2d)
    SNRdb_x_2d = np.copy(SNR_x_2d)
    SNRdb_y_2d = np.copy(SNRdb_x_2d)
    SNRdb_z_2d = np.copy(SNRdb_x_2d)
    R_xx_2d = np.copy(SNRdb_x_2d)
    R_yy_2d = np.copy(R_xx_2d)
    R_zz_2d = np.copy(R_xx_2d)
    fMSE_xx_2d = np.copy(SNRdb_x_2d)
    fMSE_yy_2d = np.copy(SNRdb_y_2d)
    fMSE_zz_2d = np.copy(SNRdb_z_2d)

    for i in range(index1.size):
        S1 = D1[index1[i], index2[i], :]
        S2 = D2[index1[i], index2[i], :]
        S3 = D3[index1[i], index2[i], :]
        DATA = np.column_stack((S1, S2, S3))

        if REF is not None:
            ref = REF[index1[i], index2[i], :]
            DATA = np.column_stack((DATA, ref))

        iok = np.isnan(np.sum(DATA, axis=1))
        tmp = np.sum(DATA, axis=1)
        iok = (~np.isnan(tmp))

        if DATA[iok, :].shape[0] == 0:
            continue

        if REF is None:
            X = DATA[iok, 0]
            Y = DATA[iok, 1]
            Z = DATA[iok, 2]
        else:
            inok = np.isnan(tmp)
            DATA[inok, :] = np.nan
            X = CDF_match(np.column_stack((DATA[iok, 4], DATA[iok, 0])))
            Y = CDF_match(np.column_stack((DATA[iok, 4], DATA[iok, 1])))
            Z = CDF_match(np.column_stack((DATA[iok, 4], DATA[iok, 2])))

        L = len(X)
        X = X.reshape((L, 1))
        Y = Y.reshape((L, 1))
        Z = Z.reshape((L, 1))

        RR = np.corrcoef(np.column_stack((X, Y, Z)).T)
        values = RR[np.ix_([0, 1, 2], [0, 1, 2])]  # Corrected indexing

        if L < nod_th or np.any(values < corr_th):
            VAR_xerr = -1
            VAR_yerr = -1
            VAR_zerr = -1
            exitflag = 0
        else:
            c1 = 1
            c2 = (np.dot(X.T, Z) / np.dot(Y.T, Z)).item()
            c3 = (np.dot(X.T, Y) / np.dot(Z.T, Y)).item()
            X = c1 * X
            Y = c2 * Y
            Z = c3 * Z

            covM = np.cov(np.column_stack((X, Y, Z)).T, ddof=0)
            XX = covM[0, 0]
            YY = covM[1, 1]
            ZZ = covM[2, 2]
            XY = covM[0, 1]
            XZ = covM[0, 2]
            YZ = covM[1, 2]

            VAR_xerr = XX - XY * XZ / YZ
            VAR_yerr = YY - XY * YZ / XZ
            VAR_zerr = ZZ - XZ * YZ / XY

            exitflag = 1

        soln = [VAR_xerr, VAR_yerr, VAR_zerr]

        if any(val < 0 for val in soln):
            fMSE_x = -1
            fMSE_y = -1
            fMSE_z = -1
            R_xx = -1
            R_yy = -1
            R_zz = -1
            SNRdb_x = np.nan
            SNRdb_y = np.nan
            SNRdb_z = np.nan
            SNR_x = np.nan
            SNR_y = np.nan
            SNR_z = np.nan
        else:
            NSR_x = VAR_xerr / (XY * XZ / YZ)
            SNR_x = 1 / NSR_x
            NSR_y = VAR_yerr / (XY * YZ / XZ)
            SNR_y = 1 / NSR_y
            NSR_z = VAR_zerr / (XZ * YZ / XY)
            SNR_z = 1 / NSR_z

            R_xy = 1 / ((1 + NSR_x) * (1 + NSR_y)) ** 0.5
            R_xz = 1 / ((1 + NSR_x) * (1 + NSR_z)) ** 0.5
            R_yz = 1 / ((1 + NSR_y) * (1 + NSR_z)) ** 0.5

            fMSE_x = 1 / (1 + SNR_x)
            fMSE_y = 1 / (1 + SNR_y)
            fMSE_z = 1 / (1 + SNR_z)

            R_xx = 1 / (1 + NSR_x)
            R_yy = 1 / (1 + NSR_y)
            R_zz = 1 / (1 + NSR_z)

            SNRdb_x = 10 * np.log10(SNR_x)
            SNRdb_y = 10 * np.log10(SNR_y)
            SNRdb_z = 10 * np.log10(SNR_z)

        if np.any(np.array([R_xx, R_yy, fMSE_x, fMSE_y]) < 0) or np.any(np.array([R_xx, R_yy, fMSE_x, fMSE_y]) > 1):
            R_xx = np.nan
            R_yy = np.nan
            R_zz = np.nan
            fMSE_x = np.nan
            fMSE_y = np.nan
            fMSE_z = np.nan
            SNRdb_x = np.nan
            SNRdb_y = np.nan
            SNRdb_z = np.nan
            VAR_xerr = np.nan
            VAR_yerr = np.nan
            VAR_zerr = np.nan

        VAR_err_x_2d[index1[i], index2[i]] = VAR_xerr
        VAR_err_y_2d[index1[i], index2[i]] = VAR_yerr
        VAR_err_z_2d[index1[i], index2[i]] = VAR_zerr

        SNR_x_2d[index1[i], index2[i]] = SNR_x
        SNR_y_2d[index1[i], index2[i]] = SNR_y
        SNR_z_2d[index1[i], index2[i]] = SNR_z

        SNRdb_x_2d[index1[i], index2[i]] = SNRdb_x
        SNRdb_y_2d[index1[i], index2[i]] = SNRdb_y
        SNRdb_z_2d[index1[i], index2[i]] = SNRdb_z

        R_xx_2d[index1[i], index2[i]] = R_xx
        R_yy_2d[index1[i], index2[i]] = R_yy
        R_zz_2d[index1[i], index2[i]] = R_zz

        fMSE_xx_2d[index1[i], index2[i]] = fMSE_x
        fMSE_yy_2d[index1[i], index2[i]] = fMSE_y
        fMSE_zz_2d[index1[i], index2[i]] = fMSE_z

    VAR_err = {'x': VAR_err_x_2d, 'y': VAR_err_y_2d, 'z': VAR_err_z_2d}
    SNR = {'x': SNR_x_2d, 'y': SNR_y_2d, 'z': SNR_z_2d}
    SNRdb = {'x': SNRdb_x_2d, 'y': SNRdb_y_2d, 'z': SNRdb_z_2d}
    R = {'x': R_xx_2d, 'y': R_yy_2d, 'z': R_zz_2d}
    fMSE = {'x': fMSE_xx_2d, 'y': fMSE_yy_2d, 'z': fMSE_zz_2d}

    return VAR_err, SNR, SNRdb, R, fMSE

def TCA_vec_old(X, Y, Z, nod_th=30, corr_th=0):

    # 0. check the NaN and fill with NaN if any of X,Y, and Z value is nan.
    combined_nan_mask = np.isnan(X) | np.isnan(Y) | np.isnan(Z)
    X[combined_nan_mask] = np.nan
    Y[combined_nan_mask] = np.nan
    Z[combined_nan_mask] = np.nan
    
    # 1. calculation for the originial data
    # Calculate covariance
    cov_corr_results = hVec.cov_corr_three(X, Y, Z)
    covXY = cov_corr_results.get('covXY')
    covXZ = cov_corr_results.get('covXZ')
    covYZ = cov_corr_results.get('covYZ')
    
    corrXY = cov_corr_results.get('corrXY')
    corrXZ = cov_corr_results.get('corrXZ')
    corrYZ = cov_corr_results.get('corrYZ')
    
    # 2. Vectorized TC calculation
    # Scale the data
    c1 = 1
    c2 = np.nansum(X*Z, axis=2) / np.nansum(Y*Z, axis=2)
    c3 = np.nansum(X*Y, axis=2) / np.nansum(Z*Y, axis=2)
    
    Xs = c1 * X
    Ys = np.expand_dims(c2, axis=2) * Y
    Zs = np.expand_dims(c3, axis=2) * Z
    X=[]; Y=[]; Z=[];
    
    # Calculate covariance with scaled Xs, Ys, and Zs
    cov_corr_results_s = hVec.cov_corr_three(Xs, Ys, Zs)
    
    covXXs = cov_corr_results_s.get('covXX')
    covYYs = cov_corr_results_s.get('covYY')
    covZZs = cov_corr_results_s.get('covZZ')
    covXYs = cov_corr_results_s.get('covXY')
    covXZs = cov_corr_results_s.get('covXZ')
    covYZs = cov_corr_results_s.get('covYZ')
    
    # Calculate correlation with scaled Xs, Ys, and Zs
    corrXYs = cov_corr_results_s.get('corrXYs')
    corrXZs = cov_corr_results_s.get('corrXZs')
    corrYZs = cov_corr_results_s.get('corrYZs')
    
    var_Xserr = covXXs - covXYs*covXZs/covYZs
    var_Yserr = covYYs - covXYs*covYZs/covXZs
    var_Zserr = covZZs - covXZs*covYZs/covXYs
    
    # Calcuate TC numbers
    SNR_Xs =  (covXYs * covXZs / covYZs) / var_Xserr
    SNR_Ys =  (covXYs * covYZs / covXZs) / var_Yserr
    SNR_Zs =  (covXZs * covYZs / covXYs) / var_Zserr
    
    R_XYs = 1 / ((1 + 1/SNR_Xs) * (1 + 1/SNR_Ys)) ** 0.5
    R_XZs = 1 / ((1 + 1/SNR_Xs) * (1 + 1/SNR_Zs)) ** 0.5
    R_YZs = 1 / ((1 + 1/SNR_Ys) * (1 + 1/SNR_Zs)) ** 0.5
    
    fMSE_Xs = 1 / (1 + SNR_Xs)
    fMSE_Ys = 1 / (1 + SNR_Ys)
    fMSE_Zs = 1 / (1 + SNR_Zs)
    
    R_XXs = 1 / (1 + 1/SNR_Xs)
    R_YYs = 1 / (1 + 1/SNR_Ys)
    R_ZZs = 1 / (1 + 1/SNR_Zs)
    
    SNRdb_Xs = 10 * np.log10(SNR_Xs)
    SNRdb_Ys = 10 * np.log10(SNR_Ys)
    SNRdb_Zs = 10 * np.log10(SNR_Zs)
    
    VAR_err = {'x': var_Xserr, 'y': var_Yserr, 'z': var_Zserr}
    SNR = {'x': SNR_Xs, 'y': SNR_Ys, 'z': SNR_Zs}
    SNRdb = {'x': SNRdb_Xs, 'y': SNRdb_Ys, 'z': SNRdb_Zs}
    R = {'x': R_XXs, 'y': R_YYs, 'z': R_ZZs}
    fMSE = {'x': fMSE_Xs, 'y': fMSE_Ys, 'z': fMSE_Zs}
    
    # 3. set the flags
    # flag on non-scaled data
    condition_corr = (corrXY < corr_th) | (corrXZ < corr_th) | (corrYZ < corr_th) #flag 1
    
    # falg on scaled data
    condition_n_valid = np.sum(~np.isnan(Xs) & ~np.isnan(Ys) & ~np.isnan(Zs), axis=2) < nod_th #flag 2
    condition_fMSE = (fMSE_Xs < 0) | (fMSE_Ys < 0) | (fMSE_Zs < 0) | (fMSE_Xs > 1) | (fMSE_Ys > 1) | (fMSE_Zs > 1) #flag 3
    condition_negative_vars_err  = (var_Xserr < 0) | (var_Yserr < 0) | (var_Zserr < 0) #flag 4
    
    flags = {'condition_corr': condition_corr, 
            'condition_n_valid': condition_n_valid,
            'condition_fMSE': condition_fMSE,
            'condition_negative_vars_err': condition_negative_vars_err}

    return VAR_err, SNR, SNRdb, R, fMSE, flags

def TCA_vec(X, Y, Z, nod_th=30, corr_th=0):
    # 0. Check for NaNs and ensure that any NaNs are consistent across datasets
    combined_nan_mask = np.isnan(X) | np.isnan(Y) | np.isnan(Z)
    X[combined_nan_mask] = np.nan
    Y[combined_nan_mask] = np.nan
    Z[combined_nan_mask] = np.nan

    # Remove completely NaN slices (if data is 3D)
    valid_mask = ~np.isnan(X).all(axis=2) & ~np.isnan(Y).all(axis=2) & ~np.isnan(Z).all(axis=2)
    X = X[valid_mask]
    Y = Y[valid_mask]
    Z = Z[valid_mask]

    # 1. Flatten the data to 2D arrays if necessary
    X_flat = X.reshape(-1, X.shape[-1])
    Y_flat = Y.reshape(-1, Y.shape[-1])
    Z_flat = Z.reshape(-1, Z.shape[-1])

    # Remove rows with NaNs
    valid_rows = ~np.isnan(X_flat).any(axis=1) & ~np.isnan(Y_flat).any(axis=1) & ~np.isnan(Z_flat).any(axis=1)
    X_flat = X_flat[valid_rows]
    Y_flat = Y_flat[valid_rows]
    Z_flat = Z_flat[valid_rows]

    # Combine the data into a single array
    data = np.stack((X_flat, Y_flat, Z_flat), axis=1)  # Shape: (N, 3, T)

    # Initialize arrays to store results
    N = data.shape[0]
    errVar_ETC = np.zeros((N, 3))
    rho2_ETC = np.zeros((N, 3))
    SNR = np.zeros((N, 3))
    SNRdb = np.zeros((N, 3))

    # Iterate over each time series (row)
    for i in range(N):
        y = data[i, :, :].T  # Shape: (T, 3)

        # Check that there are enough valid observations
        if y.shape[0] < nod_th:
            continue  # Skip if not enough data

        # Remove any remaining NaNs
        if np.isnan(y).any():
            continue  # Skip if there are NaNs

        # Compute covariance matrix
        Q_hat = np.cov(y, rowvar=False)

        # Ensure covariance matrix is full rank
        if np.linalg.matrix_rank(Q_hat) < 3:
            continue  # Skip if covariance matrix is singular

        # Calculate correlation coefficients
        try:
            rho_ETC = np.zeros(3)
            rho_ETC[0] = np.sqrt(Q_hat[0,1]*Q_hat[0,2]/(Q_hat[0,0]*Q_hat[1,2]))
            rho_ETC[1] = np.sign(Q_hat[0,2]*Q_hat[1,2]) * np.sqrt(Q_hat[0,1]*Q_hat[1,2]/(Q_hat[1,1]*Q_hat[0,2]))
            rho_ETC[2] = np.sign(Q_hat[0,1]*Q_hat[1,2]) * np.sqrt(Q_hat[0,2]*Q_hat[1,2]/(Q_hat[2,2]*Q_hat[0,1]))

            rho2_ETC[i, :] = rho_ETC**2

            # Calculate error variances
            errVar_ETC[i, 0] = Q_hat[0,0] - Q_hat[0,1]*Q_hat[0,2]/Q_hat[1,2]
            errVar_ETC[i, 1] = Q_hat[1,1] - Q_hat[0,1]*Q_hat[1,2]/Q_hat[0,2]
            errVar_ETC[i, 2] = Q_hat[2,2] - Q_hat[0,2]*Q_hat[1,2]/Q_hat[0,1]

            # Calculate SNR
            SNR[i, :] = (rho2_ETC[i, :]) / (1 - rho2_ETC[i, :])

            # Calculate SNR in dB
            SNRdb[i, :] = 10 * np.log10(SNR[i, :])

        except Exception as e:
            # Handle any mathematical errors (e.g., division by zero)
            continue

    # Prepare outputs
    VAR_err = {
        'x': errVar_ETC[:, 0],
        'y': errVar_ETC[:, 1],
        'z': errVar_ETC[:, 2]
    }
    SNR = {
        'x': SNR[:, 0],
        'y': SNR[:, 1],
        'z': SNR[:, 2]
    }
    SNRdb = {
        'x': SNRdb[:, 0],
        'y': SNRdb[:, 1],
        'z': SNRdb[:, 2]
    }
    R = {
        'x': rho2_ETC[:, 0],
        'y': rho2_ETC[:, 1],
        'z': rho2_ETC[:, 2]
    }
    fMSE = {
        'x': 1 - rho2_ETC[:, 0],
        'y': 1 - rho2_ETC[:, 1],
        'z': 1 - rho2_ETC[:, 2]
    }

    # 3. Set the flags
    # Flag on non-scaled data (not necessary here as we didn't calculate corrXY)
    # flags on correlation coefficients
    condition_corr = (R['x'] < corr_th) | (R['y'] < corr_th) | (R['z'] < corr_th)  # flag 1
    # flag on valid number of observations
    condition_n_valid = np.sum(~np.isnan(X_flat) & ~np.isnan(Y_flat) & ~np.isnan(Z_flat), axis=1) < nod_th
    # flags on fMSE
    condition_fMSE = (fMSE['x'] < 0) | (fMSE['y'] < 0) | (fMSE['z'] < 0) | (fMSE['x'] > 1) | (fMSE['y'] > 1) | (fMSE['z'] > 1)  # flag 3
    # flag on negative error variances
    condition_negative_vars_err = (errVar_ETC < 0).any(axis=1)  # flag 4

    flags = {'condition_corr': condition_corr,
            'condition_n_valid': condition_n_valid,
            'condition_fMSE': condition_fMSE,
            'condition_negative_vars_err': condition_negative_vars_err}

    return VAR_err, SNR, SNRdb, R, fMSE, flags

def ETC(D1, D2, D3, nod_th=30, corr_th=0):
    """
    Extended Triple Collocation (ETC) is a technique for estimating the
    variance of the noise error (errVar) and correlation coefficients (rho)
    of three measurement systems (e.g., satellite, in-situ, and model-based products)
    with respect to the unknown true value of the variable being measured
    (e.g., soil moisture, wind speed).

    INPUTS
    D1, D2, D3: Arrays of observations from the three measurement systems.
    They must be of the same length, and all NaNs must be removed or handled appropriately.

    OUTPUTS
    errVar_ETC: A list of error variances [errVar_D1, errVar_D2, errVar_D3].
    rho2_ETC: A list of squared correlation coefficients [rho2_D1, rho2_D2, rho2_D3].

    REFERENCE
    McColl, K.A., J. Vogelzang, A.G. Konings, D. Entekhabi, M. Piles, A. Stoffelen (2014).
    Extended Triple Collocation: Estimating errors and correlation coefficients with respect
    to an unknown target. Geophysical Research Letters 41:6229-6236.
    """

    # Convert inputs to numpy arrays
    D1 = np.asarray(D1)
    D2 = np.asarray(D2)
    D3 = np.asarray(D3)

    # Check that all inputs have the same length
    if not (len(D1) == len(D2) == len(D3)):
        raise ValueError('Error: Input data D1, D2, D3 must be of the same length.')

    # Combine the data into a single array
    y = np.column_stack((D1, D2, D3))  # Shape: (N, 3)

    # Remove any rows with NaNs
    if np.isnan(y).any():
        y = y[~np.isnan(y).any(axis=1)]

    # Catch errors in inputs
    if y.shape[1] != 3:
        raise ValueError('Error: Input data must result in an N x 3 array after removing NaNs.')

    if y.size == 0:
        raise ValueError('Error: No data left after removing NaNs.')

    # Check that each column has non-zero variance
    if np.var(y[:, 0]) == 0 or np.var(y[:, 1]) == 0 or np.var(y[:, 2]) == 0:
        raise ValueError('Error: The sample variance of each dataset must be non-zero.')

    # Estimate covariance matrix of the three measurement systems
    Q_hat = np.cov(y, rowvar=False)

    # Compute correlation coefficients
    rho_ETC = np.zeros(3)

    try:
        rho_ETC[0] = np.sqrt(Q_hat[0, 1] * Q_hat[0, 2] / (Q_hat[0, 0] * Q_hat[1, 2]))
        rho_ETC[1] = np.sign(Q_hat[0, 2] * Q_hat[1, 2]) * np.sqrt(Q_hat[0, 1] * Q_hat[1, 2] / (Q_hat[1, 1] * Q_hat[0, 2]))
        rho_ETC[2] = np.sign(Q_hat[0, 1] * Q_hat[1, 2]) * np.sqrt(Q_hat[0, 2] * Q_hat[1, 2] / (Q_hat[2, 2] * Q_hat[0, 1]))
    except (ZeroDivisionError, FloatingPointError, ValueError):
        raise ValueError('Error: Calculation of correlation coefficients failed due to invalid covariance values.')

    rho2_ETC = rho_ETC ** 2

    # Compute error variances
    errVar_ETC = np.zeros(3)
    errVar_ETC[0] = Q_hat[0, 0] - (Q_hat[0, 1] * Q_hat[0, 2]) / Q_hat[1, 2]
    errVar_ETC[1] = Q_hat[1, 1] - (Q_hat[0, 1] * Q_hat[1, 2]) / Q_hat[0, 2]
    errVar_ETC[2] = Q_hat[2, 2] - (Q_hat[0, 2] * Q_hat[1, 2]) / Q_hat[0, 1]

    # Check for negative error variances
    if np.any(errVar_ETC < 0):
        print('Warning: At least one calculated errVar is negative. This can happen if the sample size is too small or if one of the assumptions of ETC is violated.')

    # Check for negative squared correlation coefficients
    if np.any(rho2_ETC < 0):
        print('Warning: At least one calculated squared correlation coefficient is negative. This can happen if the sample size is too small or if one of the assumptions of ETC is violated.')

    # ======================================================
    # Computer SNR
    SNR = np.zeros(3)
    SNRdb = np.zeros(3)

    SNR[0] = (rho2_ETC[0]) / (1 - rho2_ETC[0])
    SNRdb[0] = 10 * np.log10(SNR[0])
    SNR[1] = (rho2_ETC[1]) / (1 - rho2_ETC[1])
    SNRdb[1] = 10 * np.log10(SNR[1])
    SNR[2] = (rho2_ETC[2]) / (1 - rho2_ETC[2])
    SNRdb[2] = 10 * np.log10(SNR[2])

    # Prepare outputs
    VAR_err = {
        'x': errVar_ETC[0],
        'y': errVar_ETC[1],
        'z': errVar_ETC[2]
    }
    SNR = {
        'x': SNR[0],
        'y': SNR[1],
        'z': SNR[2]
    }
    SNRdb = {
        'x': SNRdb[0],
        'y': SNRdb[1],
        'z': SNRdb[2]
    }
    R = {
        'x': rho2_ETC[0],
        'y': rho2_ETC[1],
        'z': rho2_ETC[2]
    }
    fMSE = {
        'x': 1 - rho2_ETC[0],
        'y': 1 - rho2_ETC[1],
        'z': 1 - rho2_ETC[2]
    }
    # Set the flags
    # Flag on non-scaled data (not necessary here as we didn't calculate corrXY)
    # flags on correlation coefficients
    condition_corr = (R['x'] < corr_th) | (R['y'] < corr_th) | (R['z'] < corr_th)  # flag 1
    # flag on valid number of observations
    condition_n_valid = np.sum(~np.isnan(D1) & ~np.isnan(D2) & ~np.isnan(D3), axis=0) < nod_th # flag 2 (axis=0 is due to 1D array
    # flags on fMSE
    condition_fMSE = (fMSE['x'] < 0) | (fMSE['y'] < 0) | (fMSE['z'] < 0) | (fMSE['x'] > 1) | (fMSE['y'] > 1) | (fMSE['z'] > 1)  # flag 3
    # flag on negative error variances
    condition_negative_vars_err = (errVar_ETC < 0).any(axis=0)  # flag 4 (axis=0 is due to 1D array)

    flags = {'condition_corr': condition_corr,
            'condition_n_valid': condition_n_valid,
            'condition_fMSE': condition_fMSE,
            'condition_negative_vars_err': condition_negative_vars_err}

    return VAR_err, SNR, SNRdb, R, fMSE, flags
