import numpy as np

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

        if L < nod_th:
            VAR_xerr = -1
            VAR_yerr = -1
            VAR_zerr = -1
            exitflag = 1
        elif np.any(values < corr_th):
            VAR_xerr = -1
            VAR_yerr = -1
            VAR_zerr = -1
            exitflag = 2
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

            exitflag = 0

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

        exitflag_2d[index1[i], index2[i]] = exitflag

    VAR_err = {'x': VAR_err_x_2d, 'y': VAR_err_y_2d, 'z': VAR_err_z_2d}
    SNR = {'x': SNR_x_2d, 'y': SNR_y_2d, 'z': SNR_z_2d}
    SNRdb = {'x': SNRdb_x_2d, 'y': SNRdb_y_2d, 'z': SNRdb_z_2d}
    R = {'x': R_xx_2d, 'y': R_yy_2d, 'z': R_zz_2d}
    fMSE = {'x': fMSE_xx_2d, 'y': fMSE_yy_2d, 'z': fMSE_zz_2d}

    return VAR_err, SNR, SNRdb, R, fMSE, exitflag_2d
