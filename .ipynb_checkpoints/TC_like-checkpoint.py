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

def TCA_vec(X, Y, Z, nod_th=30, corr_th=0):

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
