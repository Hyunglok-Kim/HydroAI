import numpy as np
import pandas as pd
import os
from os.path import exists
import re
import sys
from scipy.stats import boxcox as bc
from datetime import datetime, timedelta
import time
# libs for probabilistic modeling
# pymc

try:
    import pymc3 as pm
    import theano.tensor as tt
    pmv = 3
except ImportError:
    import pymc as pm
    import tensorflow_probability as tfp
    tt = tfp.math
    pmv = 5
    
print("PyMC version:", pm.__version__)

import bisect

#strong data
import pickle

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.font_manager as font_manager

import arviz as az
import random
import seaborn as sns
import scipy.io
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import logging
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)

# use load_data function from Data module to do not reset cache
from hydroAI.Data import load_data
    
def bc_p_pred_integrated(lam, dsm, SM1, SM2, dt, p, asm, r, et, infilt, z, a, b, d=0, K1=0, K2=0, event_opt = 'P_dry_period'):
    
    p_hat = z*dsm + a*(SM1**b+SM2**b)*dt/2 + d*(p**K1)*(asm**K2) + r + et + infilt
    
    if event_opt != 'P_dry_period':
        p_hat = (p_hat**lam-1)/lam
    
    return p_hat

### check data
def check_ncols(input_FP, file_name):
    df = pd.read_csv(input_FP+file_name, nrows=0)
    num_cols = len(df.columns)
    return num_cols

def check_cellid(input_FP, file_name, col):
  
    columns = pd.read_csv(input_FP+file_name, nrows=1).columns

    cell_id = columns[col]
        
    return cell_id
    
def find_cellid_col(input_FP, file_name, cell_id):
    
    if not hasattr(find_cellid_col, 'cache'):
        find_cellid_col.cache = {}

    if file_name not in find_cellid_col.cache:
        find_cellid_col.cache[file_name] = pd.read_csv(input_FP+file_name, nrows=1).columns.to_numpy()
    columns = find_cellid_col.cache[file_name]
    col = np.where(columns==str(cell_id))[0]
    return col[0] if len(col) > 0 else np.nan
find_cellid_col.cache = {}
    
def extract_data_from_col(data, columns, col, scale_factor, nan_fill=False):
    
    if col <= len(columns) - 1:
        cell_id = columns[col]
        
        if nan_fill:
            val = np.nan_to_num(data[cell_id].to_numpy(), nan=0)
        else:
            val = data[cell_id].to_numpy()
        
        val = val.astype('float64')
        val[val==None] = np.nan
        return val * scale_factor, cell_id

    else:
        print("Invalid column index:", col)
        return [], []
    
def make_ind_for_TR(v_P, TR):
    lst = [0]
    # set the maximum value for the last element of the list
    max_value = len(v_P)-1
    # define the step sizes and their probabilities
    step_sizes = [TR, int(TR*(3/2)), TR*2, int(TR*5/2), TR*3, int(TR*7/2)]
    probabilities = [0.6, 0.25, 0.1, 0.03, 0.01, 0.01]

    # loop while the last element of the list is less than the maximum value
    while lst[-1] < max_value:
        # randomly select a step size from the given options
        step_size = random.choices(step_sizes, weights=probabilities)[0]
        # add the step size to the last element of the list
        next_value = lst[-1] + step_size
        # check if the next value exceeds the maximum value
        if next_value > max_value:
            # if it does, add the maximum value to the list and exit the loop
            lst.append(max_value)
            break
        # otherwise, add the next value to the list
        lst.append(next_value)
    return lst, step_sizes
        
def compute_P_event_indices(start_indices, end_indices, v_idx):
    event_indices = []
    for i in range(len(start_indices)):
        start_idx = start_indices[i]
        end_idx = end_indices[i]
        start_position = bisect.bisect_left(v_idx, start_idx)
        end_position = bisect.bisect_right(v_idx, end_idx)
        event_indices.append(v_idx[start_position:end_position])
    return event_indices

def compute_valid_event_indices(rescaled_SM, x_event_indices, event_opt):
    # find max and min SM in each event and consider them and start and end dry(wet) event
    x_valid_event_indices = []
    x_valid_events = []
    
    for i in range(len(x_event_indices)):
        event_idx = x_event_indices[i]
        if len(event_idx) >= 2:

            #print('SMAP')
            #print(rescaled_SM[event_idx])
            min_idx = event_idx[np.argmin(rescaled_SM[event_idx])]
            max_idx = event_idx[np.argmax(rescaled_SM[event_idx])]
            
            if event_opt == 'P_wet' and min_idx < max_idx:
                event_idx = np.array([min_idx, max_idx])
                x_valid_events.append(event_idx)
                x_valid_event_indices.append(i)

            elif (event_opt == 'P_dry' or event_opt == 'P_dry_period') and max_idx < min_idx:
                event_idx = np.array([max_idx, min_idx])
                x_valid_events.append(event_idx)
                x_valid_event_indices.append(i)
                
    return x_valid_event_indices, x_valid_events

def compute_filtered_event_indices(rescaled_SM, x_v_events, x_v_event_indices, P_start_indices, P_end_indices, event_opt):
        
        P_v_start_indices = [P_start_indices[i] for i in x_v_event_indices]
        P_v_end_indices   = [P_end_indices[i] for i in x_v_event_indices]

        diff_sm = [rescaled_SM[event[-1]] - rescaled_SM[event[0]] for event in x_v_events]

        if event_opt == 'P_wet':
            f_x_events = [event for i, event in enumerate(x_v_events) if diff_sm[i] > 0]
            f_P_start_indices = [event for i, event in enumerate(P_v_start_indices) if diff_sm[i] > 0]
            f_P_end_indices   = [event for i, event in enumerate(P_v_end_indices) if diff_sm[i] > 0]
        elif event_opt == 'P_dry' or event_opt == 'P_dry_period':
            f_x_events = [event for i, event in enumerate(x_v_events) if diff_sm[i] < 0]
            f_P_start_indices = [event for i, event in enumerate(P_v_start_indices) if diff_sm[i] < 0]
            f_P_end_indices   = [event for i, event in enumerate(P_v_end_indices) if diff_sm[i] < 0]

        start_indices = [event[0] for event in f_x_events]
        end_indices = [event[-1] for event in f_x_events]
        
        #if event_opt == 'P_wet' or event_opt == 'P_dry':
        P_event_start_indices = [event for event in f_P_start_indices]
        P_event_end_indices = [event for event in f_P_end_indices]
        
        #elif event_opt == 'P_dry_period':
            #P_event_start_indices = start_indices
            
            #P_event_end_indices   = end_indices
    
        return start_indices, end_indices, P_event_start_indices, P_event_end_indices

def compute_valid_event_indices_or_SM(SSM_NLDAS, start_indices, end_indices, P_event_start_indices, P_event_end_indices, event_opt):

    start_indices_or_SM = np.copy(start_indices).tolist()
    end_indices_or_SM = np.copy(end_indices).tolist()
    for i, (si, ei) in enumerate(zip(P_event_start_indices, P_event_end_indices)):
        
        event_idx = list(range(si, ei))
        #print('NLDAS')
        #print(SSM_NLDAS[event_idx])
        min_idx = event_idx[np.nanargmin(SSM_NLDAS[event_idx])]
        max_idx = event_idx[np.nanargmax(SSM_NLDAS[event_idx])]

        if event_opt == 'P_wet' and min_idx < max_idx:
            start_indices_or_SM[i] = min_idx
            end_indices_or_SM[i] = max_idx
            
        elif (event_opt == 'P_dry' or event_opt == 'P_dry_period') and max_idx < min_idx:
            start_indices_or_SM[i] = max_idx
            end_indices_or_SM[i] = min_idx

        else:
            True
    
    return start_indices_or_SM, end_indices_or_SM
    
def find_P_wetup_drydown(rescaled_SM, SSM_NLDAS, P, R, ET, event_opt, P_threshold, plot_pi=False):
    # Create mask for valid indices
    mask = (~np.isnan(rescaled_SM)) & (rescaled_SM > 0) & (rescaled_SM < 1) & (~np.isnan(SSM_NLDAS)) & (~np.isnan(P)) & (~np.isnan(R)) & (~np.isnan(ET))
    v_idx = np.where(mask)[0]
    
    # Find the start and end indices of each precipitation event
    P_start_indices = []
    P_end_indices = []
    P_start = None
    for i in range(len(P)):
        if P[i] >= P_threshold and P_start is None:
            # Start of a new event
            P_start = i
        elif P[i] < P_threshold and P_start is not None:
            # End of an event
            P_start_indices.append(P_start)
            P_end_indices.append(i)
            P_start = None

    # If the last event is still ongoing, end it at the last index
    if P_start is not None:
        P_start_indices.append(P_start)
        P_end_indices.append(len(P))

    # Create an array of the next precipitation start indices
    next_P_start_indices = []
    for i in range(len(P_start_indices)):
        next_index = i + 1
        while next_index < len(P_start_indices) and P_start_indices[next_index] <= P_end_indices[i]:
            next_index += 1
        if next_index < len(P_start_indices):
            next_P_start_indices.append(P_start_indices[next_index]-1) #-1 to should not include next P event in dry period
        else:
            next_P_start_indices.append(-1)
            
    #find wet or dry indices
    if event_opt == 'P_wet' or event_opt == 'P_dry' or event_opt == 'P_dry_period':
        
        if event_opt == 'P_wet':
            x_event_indices = compute_P_event_indices(P_start_indices, P_end_indices, v_idx)
        elif event_opt == 'P_dry' or event_opt == 'P_dry_period':
            x_event_indices = compute_P_event_indices(P_end_indices, next_P_start_indices, v_idx)
            
            #x_event_indices = compute_P_event_indices(P_start_indices, next_P_start_indices, v_idx)
            
        x_v_event_indices, x_v_events = compute_valid_event_indices(rescaled_SM, x_event_indices, event_opt)

        if event_opt == 'P_wet':
            start_indices, end_indices, P_event_start_indices, P_event_end_indices = compute_filtered_event_indices(rescaled_SM, x_v_events, x_v_event_indices, P_start_indices, P_end_indices, event_opt)
        else:
            start_indices, end_indices, P_event_start_indices, P_event_end_indices = compute_filtered_event_indices(rescaled_SM, x_v_events, x_v_event_indices, P_end_indices, next_P_start_indices, event_opt)
        start_indices_or_SM, end_indices_or_SM = compute_valid_event_indices_or_SM(SSM_NLDAS, start_indices, end_indices, P_event_start_indices, P_event_end_indices, event_opt)
        
    elif event_opt == 'P_wet_dry':
        wet_event_indices = compute_P_event_indices(P_start_indices, P_end_indices, v_idx)
        wet_v_event_indices, wet_v_events = compute_valid_event_indices(rescaled_SM, wet_event_indices, 'P_wet')
        dry_event_indices = compute_P_event_indices(P_end_indices, next_P_start_indices, v_idx)
        dry_v_event_indices, dry_v_events = compute_valid_event_indices(rescaled_SM, dry_event_indices, 'P_dry')
        
        wet_dry_v_event_indices = []
        for i in range(len(wet_event_indices)):
            wet_idx = wet_event_indices[i]
            dry_idx = dry_event_indices[i]
            if len(wet_idx) >= 2 and len(dry_idx) >= 2:
                wet_dry_v_event_indices.append(i)
        
        wet_start_indices, wet_end_indices, wet_P_event_start_indices, wet_P_event_end_indices = compute_filtered_event_indices(rescaled_SM, wet_v_events, wet_v_event_indices, P_start_indices, P_end_indices, event_opt='P_wet')
        dry_start_indices, dry_end_indices, dry_P_event_start_indices, dry_P_event_end_indices = compute_filtered_event_indices(rescaled_SM, dry_v_events, dry_v_event_indices, P_start_indices, P_end_indices, event_opt='P_dry')
        
        start_indices = np.concatenate([wet_start_indices, dry_start_indices]).astype(int)
        end_indices   = np.concatenate([wet_end_indices,   dry_end_indices]).astype(int)
        P_event_start_indices = np.concatenate([wet_P_event_start_indices, dry_P_event_start_indices]).astype(int)
        P_event_end_indices   = np.concatenate([wet_P_event_end_indices,   dry_P_event_end_indices]).astype(int)

        #we don't actually need this to calcuate something, but for the consistency of the code.
        start_indices_or_SM = [] #start_indices
        end_indices_or_SM   = [] #end_indices
        
    if plot_pi is not False and event_opt == 'P_wet_dry':
        plot_P_event_both_wetup_drydown(plot_pi, SSM_NLDAS, P, R, ET, P_start_indices, P_end_indices, wet_event_indices, dry_event_indices, wet_dry_v_event_indices, offset=30, maxnlocator=5)
    
    elif plot_pi is not False and (event_opt == 'P_wet' or event_opt == 'P_dry' or event_opt == 'P_dry_period'):
        plot_P_event_wetup_drydown(plot_pi, SSM_NLDAS, P, R, ET, start_indices, end_indices, P_event_start_indices, P_event_end_indices, start_indices_or_SM, end_indices_or_SM, offset=30, maxnlocator=5)
   
    return start_indices, end_indices, P_event_start_indices, P_event_end_indices, start_indices_or_SM, end_indices_or_SM
    
def make_df_for_P_event(rescaled_SM, SSM_NLDAS, P, R, ET, JDATES, event_opt, P_threshold, plot_pi=False):

    start_indices, end_indices, P_start_indices, P_end_indices, start_indices_or_SM, end_indices_or_SM = find_P_wetup_drydown(rescaled_SM, SSM_NLDAS, P, R, ET, event_opt, P_threshold, plot_pi)

    dsm    = np.array([rescaled_SM[end] - rescaled_SM[start] for start, end in zip(start_indices, end_indices)])
    dsm_or = np.array([SSM_NLDAS[end]   - SSM_NLDAS[start]  for start, end in zip(start_indices_or_SM, end_indices_or_SM)])
    
    if event_opt == 'P_wet' or event_opt == 'P_dry':
        dsm    = np.abs(dsm)
        dsm_or = np.abs(dsm_or)
        
    # Calculate statistics for filtered intervals
    sumP  = np.array([np.sum(P[start:end + 1]) for start, end in zip(P_start_indices, P_end_indices)])    
    sumR  = np.array([np.sum(R[start:end + 1]) for start, end in zip(P_start_indices, P_end_indices)])
    sumET = np.array([np.sum(ET[start:end + 1]) for start, end in zip(P_start_indices, P_end_indices)])
    dt    = (JDATES[end_indices] - JDATES[start_indices]).astype("timedelta64[h]").astype(np.float64)

    # Create DataFrame with calculated statistics
    df = pd.DataFrame({
        't1': JDATES[start_indices],
        't2': JDATES[end_indices],
        'SM1': rescaled_SM[start_indices],
        'SM2': rescaled_SM[end_indices],
        'dSM': dsm,
        'dt': dt,
        'SM1_or': SSM_NLDAS[start_indices_or_SM],
        'SM2_or': SSM_NLDAS[end_indices_or_SM],
        'dSM_or': dsm_or,
        'dt': dt,
        'sumP': sumP,
        'sumR': sumR,
        'sumET': sumET,
        'start_idx': start_indices,
        'end_idx': end_indices,
        'P_start_idx': P_start_indices,
        'P_end_idx': P_end_indices
    })

    df.sort_values(by='t1', inplace=True)

    return df

def find_wetup(rescaled_SM, P, R, ET, case, P_threshold, threshold_condition=1, dssm_th = 0):
    
    #threshold_condition 1: lose - summation of P > P_threshold. More data/less accurate.
    #threshold_condition 2: strict - during the wet period, all P should > P_threshold
    
    # Create mask for valid indices
    mask = (~np.isnan(rescaled_SM)) & (rescaled_SM > 0) & (rescaled_SM < 1) & (~np.isnan(P)) & (~np.isnan(R)) & (~np.isnan(ET))
    v_idx = np.where(mask)[0]

    # Extract rescaled SM and JDATES for valid indices
    v_ssm    = rescaled_SM[v_idx]
    
    # Identify start and end indices of increasing v_ssm
    v_start_indices = []
    v_end_indices = []
    for i in range(len(v_ssm) - 1):
        if v_ssm[i] < v_ssm[i + 1] and (i == 0 or v_ssm[i - 1] >= v_ssm[i]):
            v_start_indices.append(i)
        if v_ssm[i] < (v_ssm[i + 1] - dssm_th) and (i == len(v_ssm) - 2 or v_ssm[i + 1] >= (v_ssm[i + 2] + dssm_th)):
            v_end_indices.append(i + 1)
        
    if len(v_end_indices) > 0 and v_end_indices[-1] < v_start_indices[-1]:
        v_end_indices.append(len(v_ssm) - 1)

    # Convert relative indices to original indices
    start_indices = v_idx[v_start_indices]
    end_indices = v_idx[v_end_indices]

    # Filter start and end indices based on P_threshold
    wetup_start_indices = []
    wetup_end_indices = []
    for s, e in zip(start_indices, end_indices):
        p_values = P[s:e+1]
        
        if threshold_condition==1:
            th_con = np.sum(p_values)>=P_threshold
        else:
            th_con = np.all(p_values >= P_threshold)
        if th_con: 
            wetup_start_indices.append(s)
            wetup_end_indices.append(e)

    #find previous P event that possibley makes the soil wet previously
    P_event_start_indices = wetup_start_indices.copy()
    
    # I think wetup period does not need to account previous P that does not include in the wetup period
    #P_event_end_indices = wetup_start_indices.copy()
    #for i in range(len(wetup_start_indices)):
    #    start_index = wetup_start_indices[i]

    #    # loop through the previous indices until the precipitation value is larger than P_threshold
    #    for j in range(1, start_index + 1):
    #        if P[start_index - j] > P_threshold:
    #            P_event_end_indices[i] = (start_index - j)
    #            break

    #    P_event_end_index = P_event_end_indices[i]
    #    for j in range(1, P_event_end_index + 1):
    #        if P[P_event_end_index - j] <= P_threshold:
    #            P_event_start_indices[i] = P_event_end_index - j
    #            break
    
    return wetup_start_indices, wetup_end_indices, P_event_start_indices

def find_drydown(rescaled_SM, P, R, ET, case, P_threshold=0.01):
    # Create mask for valid indices
    mask = (~np.isnan(rescaled_SM)) & (rescaled_SM > 0) & (rescaled_SM < 1) & (~np.isnan(P)) & (~np.isnan(R)) & (~np.isnan(ET))
    v_idx = np.where(mask)[0]

    # Extract rescaled SM and JDATES for valid indices
    v_ssm = rescaled_SM[v_idx]

    # Identify start and end indices of increasing v_ssm
    v_start_indices = []
    v_end_indices = []
    for i in range(len(v_ssm) - 1):
        if v_ssm[i] > v_ssm[i + 1] and (i == 0 or v_ssm[i - 1] <= v_ssm[i]):
            v_start_indices.append(i)
        if v_ssm[i] > v_ssm[i + 1] and (i == len(v_ssm) - 2 or v_ssm[i + 1] <= v_ssm[i + 2]):
            v_end_indices.append(i + 1)
    if len(v_end_indices) > 0 and v_end_indices[-1] < v_start_indices[-1]:
        v_end_indices.append(len(v_ssm) - 1)

    # Convert relative indices to original indices
    start_indices = v_idx[v_start_indices]
    end_indices = v_idx[v_end_indices]

    # Filter start and end indices based on P_threshold
    drydown_start_indices = []
    drydown_end_indices = []
    for s, e in zip(start_indices, end_indices):
        p_values = P[s:e+1]
        if np.all(p_values <= P_threshold):
            drydown_start_indices.append(s)
            drydown_end_indices.append(e)

    #find previous P event that possibley makes the soil wet previously
    P_event_start_indices = drydown_start_indices.copy()
    P_event_end_indices = drydown_start_indices.copy()
    for i in range(len(drydown_start_indices)):
        start_index = drydown_start_indices[i]

        # loop through the previous indices until the precipitation value is larger than P_threshold
        for j in range(1, start_index + 1):
            if P[start_index - j] > P_threshold:
                P_event_end_indices[i] = (start_index - j)
                break

        P_event_end_index = P_event_end_indices[i]
        for j in range(1, P_event_end_index + 1):
            if P[P_event_end_index - j] <= P_threshold:
                P_event_start_indices[i] = P_event_end_index - j
                break
    
    return drydown_start_indices, drydown_end_indices, P_event_start_indices

def make_df_for_event(rescaled_SM, rescaled_SM_or, P, R, ET, JDATES, case, P_threshold=0.01, event_opt='wet'):

    if event_opt == 'wet':
        start_indices, end_indices, P_start_indices = find_wetup(rescaled_SM, P, R, ET, case, P_threshold)
        dsm    = np.array([rescaled_SM[end]    - rescaled_SM[start]    for start, end in zip(start_indices, end_indices)])
        dsm_or = np.array([rescaled_SM_or[end] - rescaled_SM_or[start] for start, end in zip(start_indices, end_indices)])
    elif event_opt == 'dry':
        start_indices, end_indices, P_start_indices = find_drydown(rescaled_SM, P, R, ET, case, P_threshold)
        dsm    = np.array([rescaled_SM[start]    - rescaled_SM[end]    for start, end in zip(start_indices, end_indices)])
        dsm_or = np.array([rescaled_SM_or[start] - rescaled_SM_or[end] for start, end in zip(start_indices, end_indices)])
        
    # Calculate statistics for filtered intervals
    sumP = np.array([np.sum(P[start:end + 1]) for start, end in zip(P_start_indices, end_indices)])    
    sumR = np.array([np.sum(R[start:end + 1]) for start, end in zip(P_start_indices, end_indices)])      
    sumET = np.array([np.sum(ET[start:end + 1]) for start, end in zip(P_start_indices, end_indices)])
    dt = (JDATES[end_indices] - JDATES[start_indices]).astype("timedelta64[h]").astype(np.float64)
                 
    # Create DataFrame with calculated statistics
    df = pd.DataFrame({
        't1': JDATES[start_indices],
        't2': JDATES[end_indices],
        'SM1': rescaled_SM[start_indices],
        'SM2': rescaled_SM[end_indices],
        'dSM': dsm,
        'dt': dt,
        'SM1_or': rescaled_SM_or[start_indices],
        'SM2_or': rescaled_SM_or[end_indices],
        'dSM_or': dsm_or,
        'sumP': sumP,
        'sumR': sumR,
        'sumET': sumET,
        'start_idx': start_indices,
        'end_idx': end_indices,
        'P_start_idx': P_start_indices,
        'P_end_idx': end_indices
    })
                 
    df.sort_values(by='t1', inplace=True)

    return df

def rescale_SM(SSM_SMAPL3, SSM_NLDAS, P, TR_argument):
    
    TR = TR_argument[0]
    
    if TR == 'like':

        rescale_SSM = np.zeros((len(SSM_NLDAS)))
        ind         = np.argwhere(~np.isnan(SSM_SMAPL3) & (SSM_SMAPL3 > 0)).reshape(-1,)

        rescale_SSM[ind]                           = SSM_NLDAS[ind]
        rescale_SSM[np.argwhere(rescale_SSM == 0)] = np.nan

        SSM_save = [rescale_SSM.copy()]  

        TR_it       = 1
        masking_day = 5
        sample_rate = 1

    elif TR == 'SMAPL3':
        SSM_save         = [SSM_SMAPL3.copy()]

        TR_it       = 1
        masking_day = 5
        sample_rate = 1

    else:
        v_SSM_NLDAS = SSM_NLDAS.copy()

        trit        = TR_argument[1]
        diversity   = TR_argument[2]
        sample_rate = TR_argument[3]
        rescale_SSM_save = [np.array([]) for _ in range(TR)]

        TR_it = 1 
        if trit == 1:
            TR_it = TR 

        for i in range(TR_it):
            rescale_SSM = np.zeros((len(v_SSM_NLDAS)))
            first_valid_point = int(np.argwhere(~np.isnan(v_SSM_NLDAS))[0])
            ind               = list(range(first_valid_point, len(v_SSM_NLDAS)-1, TR))
            masking_day       = TR/24
            if diversity == 1:
                ind, step_sizes = make_ind_for_TR(P, TR)
                masking_day = step_sizes[-1]/24

            rescale_SSM[ind]                           = v_SSM_NLDAS[ind]
            rescale_SSM[np.argwhere(rescale_SSM == 0)] = np.nan
            rescale_SSM_save[i]                        = rescale_SSM

            v_SSM_NLDAS[first_valid_point] = np.nan

        SSM_save = rescale_SSM_save.copy()
    
    return SSM_save, SSM_NLDAS, TR_it, masking_day, sample_rate

def make_df(SSM_SMAPL3, SSM_NLDAS, P, R, ET, case, TR_argument, GN_std, input_FP, file_names, dth, P_threshold, event_opt='wet'):
    
    JDATES = pd.date_range(start='2015-01-01 00:30:00', end='2021-12-31 23:30:00', freq='1h').values.reshape(-1,)
   
    TR = TR_argument[0]
    
    # rescale SM
    SSM_save, noscale_SSM_NLDAS, TR_it, masking_day, sample_rate = rescale_SM(SSM_SMAPL3, SSM_NLDAS, P, TR_argument)
    #SSM_save_or = rescale_SM(SSM_SMAPL3, SSM_NLDAS, P, TR_argument)[0]

    # add Gaussian noise
    for i in range(TR_it):
        GN                          = np.random.normal(0, GN_std, sum(~np.isnan(SSM_save[i])))
        valid_point                 = np.argwhere(~np.isnan(SSM_save[i]))
        SSM_save[i][valid_point]    = SSM_save[i][valid_point] + GN.reshape(-1,1)

    column_names = ['t1', 't2', 'SM1', 
                    'SM2', 'dSM', 'dt',
                    'SM1_or', 'SM2_or', 'dSM_or',
                    'sumP',  'sumR', 'sumET',
                    'start_idx','end_idx',
                    'P_start_idx', 'TR_it_id']
    df = pd.DataFrame(columns=column_names)
    
    for ii in range(TR_it):
        
        if event_opt == 'wet' or event_opt == 'dry':
            t_df    = make_df_for_event(SSM_save[ii], SSM_NLDAS, P, R, ET, JDATES, case, P_threshold=P_threshold, event_opt=event_opt)
            t_df['TR_it_id']    = ii
            masking_day = 99999
            
        elif event_opt == 'all':
            t_df_wet = make_df_for_event(SSM_save[ii], SSM_NLDAS, P, R, ET, JDATES, case, P_threshold=P_threshold, event_opt='wet')
            t_df_dry = make_df_for_event(SSM_save[ii], SSM_NLDAS, P, R, ET, JDATES, case, P_threshold=P_threshold, event_opt='dry')
            t_df     = pd.concat([t_df_wet, t_df_dry], ignore_index=True)
            t_df['TR_it_id'] = ii
            
            masking_day = 99999
        
        elif event_opt == 'P_wet' or event_opt == 'P_dry' or event_opt == 'P_wet_dry' or event_opt == 'P_dry_period':
            t_df = make_df_for_P_event(SSM_save[ii], SSM_NLDAS, P, R, ET, JDATES, event_opt, P_threshold=P_threshold, plot_pi=False)
            t_df['TR_it_id'] = ii
             
            masking_day = 99999
            
        else:
            mask = (~np.isnan(SSM_save[ii])) & (SSM_save[ii] > 0) & (SSM_save[ii] < 1) & (~np.isnan(P)) & (~np.isnan(R)) & (~np.isnan(ET))
            v_idx = np.where(mask)[0]
            
            v_ssm    = SSM_save[ii][v_idx]
            v_jdates = JDATES[v_idx]
            dt       = np.float32((v_jdates[1:] - v_jdates[:-1]).astype('timedelta64[h]'))
            sumP     = np.add.reduceat(P, v_idx)[:-1] # mm/dt; by doing this it is already mm/dt unit
            sumR     = np.add.reduceat(R, v_idx)[:-1] # mm/dt; by doing this it is already mm/dt unit
            sumET    = np.add.reduceat(ET, v_idx)[:-1] # mm/dt; by doing this it is already mm/dt unit
            dssm     = np.diff(v_ssm) # (mm/mm)/dt; it is already (mm/mm)/dt unit/ dt is not fixed
            dssm_idx = np.argwhere(dssm > dth).reshape(-1,)
            
            v_ssm_or = SSM_save_or[ii][v_idx]
            dssm_or  = np.diff(v_ssm_or) # (mm/mm)/dt; it is already (mm/mm)/dt unit/ dt is not fixed
            
            t_df = pd.DataFrame(columns = column_names)
            for i, t_dsm_idx in enumerate(dssm_idx):

                t_t1   = v_jdates[t_dsm_idx]
                t_t2   = v_jdates[t_dsm_idx+1]
                t_SM1  = v_ssm[t_dsm_idx]
                t_SM2  = v_ssm[t_dsm_idx+1]
                t_dSM  = t_SM2 - t_SM1
                t_dt   = np.float32((t_t2 - t_t1).astype('timedelta64[h]'))

                t_SM1_or  = v_ssm_or[t_dsm_idx]
                t_SM2_or  = v_ssm_or[t_dsm_idx+1]
                t_dSM_or  = t_SM2_or - t_SM1_or
                
                t_sumP = sumP[t_dsm_idx]
                t_sumR = sumR[t_dsm_idx]
                t_sumET = sumET[t_dsm_idx]
                
                start_idx = t_dsm_idx
                end_idx = t_dsm_idx+1
                P_start_idx = t_dsm_idx
        
                t_list = [t_t1, t_t2, t_SM1, 
                          t_SM2, t_dSM, t_dt,
                          t_SM1_or, t_SM2_or, t_dSM_or,
                          t_sumP, t_sumR, t_sumET, start_idx, end_idx,
                          P_start_idx, ii]
      
                tt_df = pd.DataFrame([t_list], columns=column_names)
                t_df  = pd.concat([t_df, tt_df], axis=0)

        df = pd.concat([df, t_df], axis=0, ignore_index=True)

    #df = df[df.sumP > P_threshold]

    if event_opt != 'P_dry_period':
            res = df.sumP - 100*np.abs(df.dSM_or) - (df.sumR + df.sumET)
    else:
            res = 100*np.abs(df.dSM_or) + df.sumP - (df.sumR+df.sumET)
    df['res'] = res
    
    df = df.dropna(how='any')
    # mask Jan and Dec
    df['t1'] = pd.to_datetime(df['t1'])
    mask_date = ~(df['t1'].dt.month.isin([1,12]))
    df = df.loc[mask_date]
    mask = df.dt<=24*masking_day
    df = df.loc[mask].sample(frac=sample_rate)
    df.sort_values(by='t1', inplace=True)
    df = df[df.dt<1000] # more than 1000 hrs is unrealistic
    df = df.drop_duplicates()
    
    df = df.reset_index(drop=True)
    return df, SSM_save, P, R, noscale_SSM_NLDAS

def make_input(df, case):
    p       = pd.to_numeric(df.sumP).values # mm/dt
    SM1     = pd.to_numeric(df.SM1).values
    SM2     = pd.to_numeric(df.SM2).values
    dsm     = pd.to_numeric((df.dSM)).values
    dt      = pd.to_numeric(df.dt).values
    SM1_or  = pd.to_numeric(df.SM1_or).values
    SM2_or  = pd.to_numeric(df.SM2_or).values
    dsm_or  = pd.to_numeric((df.dSM_or)).values
    r       = pd.to_numeric(df.sumR).values # mm/dt
    et      = pd.to_numeric(df.sumET).values # mm/dt
    infilt  = 0
    asm     = 0
    res     = pd.to_numeric(df.res).values # mm/dt
    
    #if (case == x) | (case == x):
    #    asm    = pd.to_numeric(df.SM1).values

    return p, dsm, dsm_or, SM1, SM2, dt, r, et, asm, infilt, res

def f_ET_I(SM1, SM2, a, b, dt):
    ET_I_est = a*(SM1**b + SM2**b)*dt/2
    return ET_I_est
    
def f_R(asm, p, d, K1, K2):
    R_est = d*(p**K1)*(asm**K2)
    return R_est

if pmv == 3:
    def logp_zero_inflated_exponential(value, psi, mu):
        return tt.switch(tt.eq(value, 0), tt.log(psi), tt.log1p(-psi) - mu * value)

if pmv == 5:
    from pytensor.tensor import TensorVariable
    def logp_zero_inflated_exponential(value: TensorVariable, psi: TensorVariable, mu: TensorVariable) -> TensorVariable:
      if value == 0:
        return np.log(psi)
      else:
        return np.log1p(-psi) - mu * value
          
# Infilt might be an incorrect term
# Instead, the percoloation term may be used

def make_idata(p, dsm, dsm_or, SM1, SM2, dt, r, et, asm, infilt, res, case, method='advi', event_opt='P_dry_period'):
    idata = 0
    valid = 0
    offset = 0
    if p.shape[0] > 30:
        
        n_draws = 1000
        n_tune  = 1000
        with pm.Model() as model:
            # ET + D = a*avgSM^b*dt
            #a  = pm.HalfNormal('α', sigma=100) # weak informative prior (we know it should be positive) 
            beta_hyper = pm.HalfCauchy('beta_hyper', beta=1)
            a  = pm.HalfCauchy('α', beta=beta_hyper)
            if pmv == 3:
                b  = pm.Normal('β', mu=0, sigma=1, testval=0)
            if pmv == 5:
                b  = pm.Normal('β', mu=0, sigma=1, initval=0)

            # Q ~ d*(P^K1)*(SM^K2)
            #d  = 0
            #K1 = 0
            #K2 = 0
              
            #if case == 4:  should be corrected later
            #    d  = pm.HalfNormal('δ', sigma=1000, initval=1)#, initval=0.5)
            #    K1 = 1
            #    K2 = 1

            #if case == 5: should be corrected later
            #    d  = pm.HalfNormal('δ', sigma=1000, initval=1)#, initval=0.5)
            #    K1 = pm.HalfNormal('K1', sigma=1000, initval=1) #, initval=0.5)
            #    K2 = K1

            sd = pm.HalfNormal('sd', sigma=1)
            if event_opt != 'P_dry_period': #wet-up based calculation
                if case == 1:
                    #####
                    # With the case-1, the summation of p is ummation of z*|dSM| and ET+I
                    # P = z*|dSM| + ET + I
                    # This case, we use the estimated ET+I from f_et_i(SM).
                    # p = z*|dSM| + f_et_i(SM)
                    # However, in this case, if P = 0, we cannot use the lognormal for the likelihood fn.
                    # Thereroe, we add the offset factor 1e-10 to P.
                    #####
                    #z  = pm.HalfNormal('Z', sigma=500)
                    z = pm.TruncatedNormal('Z', mu=100, sigma=100, lower=0, upper=500)
                    offset   = -1e-10
                    y_obs    = np.abs(dsm)
                    ET_I_est = f_ET_I(SM1, SM2, a, b, dt)
                    mu       = (p - ET_I_est) / z
                    Y_obs    = pm.Normal('Y_obs', mu=mu, sigma=sd, observed=y_obs)
                    
                if case == 2:
                    #####
                    # With the case-2, the summation of p is ummation of z*|dSM| and R+ET+I.
                    # P = z*|dSM| + ET + I + R
                    # This case, we use the estimated ET+I from f_et_i(SM), obs R.
                    # P - R = z*|dSM| + f_et_i(s)
                    # However, in this case, if P-R = 0, we cannot use the lognormal for the likelihood fn.
                    # Thereroe, we add the offset factor 1e-10 to P - R.
                    #####
                    #z  = pm.HalfNormal('Z', sigma=500)
                    z = pm.TruncatedNormal('Z', mu=100, sigma=100, lower=0, upper=500)
                    offset   = 0
                    y_obs    = np.abs(dsm)
                    ET_I_est = f_ET_I(SM1, SM2, a, b, dt)
                    mu       = (p - (r + ET_I_est))/z 
                    #mu       = (p - (r + ET_I_est + res))/z #this proves that this approach is correct!
                    Y_obs    = pm.Normal('Y_obs', mu=mu, sigma=sd, observed=y_obs)
                    
                if case == 3:
                    #####
                    # With the case-3, the summation of p is summation of z*|dSM| and R+ET+I.
                    # P = z*|dsm| + ET + I + R
                    # This case, we use both obs for R and E, and assume I=0.
                    # P - R - ET = z*|dsm|
                    # However, in this case, if P-R-ET<=0, we cannot use the lognormal for the likelihood fn.
                    # Thereroe, we add the offset factor 1e-10 to min(ET+R).               
                    #####
                    log_Z     = pm.Normal('log_Z', mu=0, sigma=1)
                    offset    = np.min(p-r-et) - 1e-10
                    y_obs     = p - r - et - offset
                    log_y_obs = np.log(y_obs)
                    log_dSM   = np.log(np.abs(dsm))

                    mu        = log_Z + log_dSM
                    Y_obs     = pm.Normal('Y_obs', mu=mu, sigma=sd, observed=log_y_obs)
                    
                if case == 4:
                    #####
                    # With the case-4, the summation of p is ummation of z*|dSM| and R+ET+I+etc.
                    # P = z*|dSM| + ET + I + R + etc.
                    # This case, we use estimated ET+I from f_et_i(SM), obs R P, and etc.
                    # z*|dSM| = P - (R + f_et_i(s) - et + res): because res term include true et.
                    # |dSM|   = (P - (R + f_et_i(SM) - et + res))/z
                    # This case shows how incorrect physcis in f_et_i affect z.
                    #####
 
                    #z  = pm.HalfNormal('Z', sigma=500)
                    z = pm.TruncatedNormal('Z', mu=100, sigma=100, lower=0, upper=500)
                    offset   = 0
                    y_obs    = np.abs(dsm)
                    ET_I_est = f_ET_I(SM1, SM2, a, b, dt)
                    mu       = (p - (r + ET_I_est + res))/z 
                    #mu       = (p - (r + ET_I_est + res))/z #this proves that this approach is correct!
                    Y_obs    = pm.Normal('Y_obs', mu=mu, sigma=sd, observed=y_obs)
                     
                if case == 'semi_true':
                    #####
                    #This is semi_true and can show why even though we do not miss constraints
                    #how the low qaulity and high TR could effect z.
                    #####
                    log_Z     = pm.Normal('log_Z', mu=0, sigma=1)
                    offset    = 0 #np.min(p - r - et - res) - 1e-10
                    y_obs     = p - r - et - res - offset
                    log_y_obs = np.log(y_obs)
                    log_dSM   = np.log(np.abs(dsm)) 
                    mu        = log_Z + log_dSM
                    Y_obs     = pm.Normal('Y_obs', mu=mu, sigma=sd, observed=log_y_obs)

                if case == 'true':
                    #####
                    #This is the true case. np.exp(log_Z) must be 100.
                    #even if we change TR or quality of data, this value should always 100.
                    #####
                    log_Z     = pm.Normal('log_Z', mu=0, sigma=1)
                    offset    = 0 #np.min(p - r - et - res) - 1e-10
                    y_obs     = p - r - et - res - offset
                    log_y_obs = np.log(y_obs)
                    log_dSM   = np.log(np.abs(dsm_or)) 
                    mu        = log_Z + log_dSM
                    Y_obs     = pm.Normal('Y_obs', mu=mu, sigma=sd, observed=log_y_obs)
                    
            else:

                if case == 1:
                    #####
                    # With the case-1, the decrease in SM is due to ET+I
                    # P = -z*|dSM| + ET + I
                    # This case, we use estimated ET+I from f_et_i(SM) and assume P=0.
                    # 0 = -z*|dSM| + f_et_i(SM)
                    # |dSM| = f_et_i(SM)/z
                    # We do not need to add the offset since we assumed the Normal dist for the likelihood fn.
                    # This case shows how missing constraints and incorrect physcis in f_et_i affect z.
                    #####
                    #z  = pm.HalfNormal('Z', sigma=1000)
                    z = pm.TruncatedNormal('Z', mu=100, sigma=100, lower=0, upper=500)
                    #beta_hyper = pm.HalfCauchy('beta_hyper', beta=1)
                    #z = pm.HalfCauchy('Z', beta=1)
                    
                    offset   = 0
                    y_obs    = np.abs(dsm)                                    
                    ET_I_est = f_ET_I(SM1, SM2, a, b, dt)
                    mu       = (ET_I_est)/z
                    Y_obs    = pm.Normal('Y_obs', mu=mu, sigma=sd, observed=y_obs)
                    
                if case == 2:
                    #####
                    # With the case-2, the decrease in SM is due to R + ET + I
                    # P = -z*|dSM| + ET + I + R
                    # This case, we use estimated ET+I from f_et_i(SM), obs R and P.
                    # z*|dSM| = R + f_et_i(s) - P
                    # |dSM| = (R + f_et_i(SM) - P)/z
                    # This case shows how missing constraints and incorrect physcis in f_et_i affect z.
                    #####
                    z  = pm.HalfNormal('Z', sigma=500)
                    offset   = 0
                    y_obs    = np.abs(dsm)                    
                    ET_I_est = f_ET_I(SM1, SM2, a, b, dt)
                    mu       = (r + ET_I_est - p)/z
                    #mu = (r + et - p + res)/z #this proves that this approach is correct!
                    Y_obs    = pm.Normal('Y_obs', mu=mu, sigma=sd, observed=y_obs)

                    ### old (pymc3 version)
                    # However, we have a lot of 0 in R data. It indicates the presence of excess zeros 
                    # that cannot be adequately explained by a standard continuous distribution. In such 
                    # cases, considering a zero-inflated exponential continuous distribution can be 
                    # beneficial.
                    # This only works with pymc3 version.
                    #psi = pm.Beta('psi', 1, 1)
                    #offset = 0# np.min(r)
                    #r[r<=0] = 0
                    #y_obs = r #-offset
                    #ET_I_est = f_ET_I(SM1, SM2, a, b, dt)
                    #mu = z*np.abs(dsm) - ET_I_est
                    ## Zero-Inflated Exponential likelihood
                    ## This only works with pymc version 3.x
                    #Y_obs = pm.DensityDist('Y_obs', logp_zero_inflated_exponential, 
                    #                       observed={'value': y_obs, 'psi': psi, 'mu': mu})
                    
                if case == 3:
                    #####
                    # With the case-3, the decrease in SM is due to R+ET+I
                    # P = -z*|dsm| + ET + I + R
                    # This case, we use both obs for R and E, and assume P=0 and I=0.
                    # 0 = -z*|dsm| + ET + R
                    # ET+R = z*|dsm|
                    # However, in this case, if R+ET<=0, we cannot use the log transform.
                    # Thereroe, we add the offset factor 1e-10 to min(ET+R).
                    # This case shows how missing constraints affects z.
                    # But, Obs ET might not represent ET from the top 10cm soil layer.
                    # So, This can can be "weak" wrong physics compared to cases 1 and 2.
                    #####
                    log_Z     = pm.Normal('log_Z', mu=0, sigma=10)
                    offset    = np.min(r+et) - 1e-10
                    y_obs     = r + et - offset
                    log_y_obs = np.log(y_obs)
                    log_dSM   = np.log(np.abs(dsm))
                    mu        = log_Z + log_dSM
                    Y_obs     = pm.Normal('Y_obs', mu=mu, sigma=sd, observed=log_y_obs)
                    
                if case == 4:
                    #####
                    # With the case-4, the decrease in SM is due to R + ET + I + etc.
                    # P = -z*|dSM| + ET + I + R + etc.
                    # This case, we use estimated ET+I from f_et_i(SM), obs R P, and etc.
                    # z*|dSM| = R + f_et_i(s) - et + res - P: because res term include true et.
                    # |dSM| = (R + f_et_i(SM) - et + res - P)/z
                    # This case shows how incorrect physcis in f_et_i affect z.
                    #####
                    #z  = pm.HalfNormal('Z', sigma=500)
                    z = pm.TruncatedNormal('Z', mu=100, sigma=100, lower=0, upper=500)
                    offset   = 0
                    y_obs    = np.abs(dsm)                    
                    ET_I_est = f_ET_I(SM1, SM2, a, b, dt)
                    mu       = (r + ET_I_est + res - p)/z
                    #mu = (r + et - p + res)/z #this proves that this approach is correct!
                    Y_obs  = pm.Normal('Y_obs', mu=mu, sigma=sd, observed=y_obs)
                    
                if case == 'semi_true':
                    #####
                    #This is semi_true and can show why even though we do not miss constraints
                    #how the low qaulity and high TR could effect z.
                    #####
                    log_Z     = pm.Normal('log_Z', mu=0, sigma=1)
                    offset    = 0#np.min(r + et + res - p) - 1e-10
                    y_obs     = r + et + res - p - offset #p - r - et - res - offset
                    log_y_obs = np.log(y_obs)
                    log_dSM   = np.log(np.abs(dsm)) 
                    
                    mu     = log_Z + log_dSM
                    Y_obs  = pm.Normal('Y_obs', mu=mu, sigma=sd, observed=log_y_obs)
                
                if case == 'true':
                    #####
                    #This is the true case. np.exp(log_Z) must be 100.
                    #even if we change TR or quality of data, this value should always 100.
                    #####
                    log_Z     = pm.Normal('log_Z', mu=0, sigma=1)
                    offset    = 0#np.min(r + et + res - p) - 1e-10
                    y_obs     = r + et + res - p - offset #p - r - et - res - offset
                    log_y_obs = np.log(y_obs)
                    log_dSM   = np.log(np.abs(dsm_or)) 
                    
                    mu     = log_Z + log_dSM
                    Y_obs  = pm.Normal('Y_obs', mu=mu, sigma=sd, observed=log_y_obs)

                    
        # sampling or fit
        rng = 321
        if method == 'nuts':
            with model:
                idata = pm.sample_prior_predictive(samples=len(p), random_seed=rng)
                idata.extend(pm.sample(draws=n_draws, tune=n_tune, random_seed=rng, 
                                       chains=2, target_accept=0.99, init="advi", progressbar=False))
                pm.sample_posterior_predictive(idata, extend_inferencedata=True, 
                                               random_seed=rng, progressbar=False)

        elif method == 'advi':
            obj_optimizer = pm.adam(learning_rate=0.01)
            with model:
                fit_advi = pm.fit(method = pm.ADVI(), n = 200000, callbacks=[pm.callbacks.CheckParametersConvergence(diff='relative')], obj_optimizer=obj_optimizer, progressbar=False)
                idata    = fit_advi.sample(2000)
        valid = 1
    else:
        print('Not enough data to fit')
    return idata, valid, offset

def save_idata(idata, valid, lam, case, method, TR, GN_std, save_dir, cell_id):
    t_file_name = cell_id+'_c_'+str(case)+'_m_'+method+'_tr_'+str(TR)+'_gn_'+str(GN_std)
    file_exists = exists(save_dir+t_file_name)

    parameter_values = []
    if valid == 0:
        print('sampling not taken: not enough data {}'.format(t_file_name))
        t_file_name = 'bad_'+t_file_name

    elif valid == 1:    

        if pmv == 3:
            if case == 3 or case == 'true':
                parameter_names = ['Z', 'α', 'β']
            else:
                parameter_names = ['log_Z', 'α', 'β']
                
            for parameter_name in parameter_names:
                values = idata.get_values(parameter_name)
                parameter_values.append({parameter_name: values})
      
        if pmv == 5:
            parameter_values = idata
            
    file = open(save_dir+t_file_name, 'wb')
    pickle.dump(valid, file)
    pickle.dump(cell_id, file)
    pickle.dump(parameter_values, file)
    pickle.dump(lam, file)
    file.close()
    
def check_file_exist(case, method, TR, GN_std, save_dir, cell_id):
    t_file_name = cell_id+'_c_'+str(case)+'_m_'+method+'_tr_'+str(TR)+'_gn_'+str(GN_std)
    file_exists = exists(save_dir+t_file_name)
    
    return file_exists

### main pymc fitting code
def fitting(SSM_SMAPL3, SSM_NLDAS, P, R, ET, cell_id, case, method, TR, GN_std, input_FP, save_dir, file_names, sub_opt, div_opt, sample_rate_opt, save_idata_opt, event_opt, p_threshold):
    
    dth = 0.001         #dSM threshold
    
    if save_idata_opt == 1:
        check_file = check_file_exist(case, method, TR, GN_std, save_dir, cell_id)
        
        if check_file:
            #print('idata data already exists')
            return 0, 0, 0, 0
        else:
            print(cell_id)
            TR_argument = [TR, sub_opt, div_opt, sample_rate_opt]
            
            df = make_df(SSM_SMAPL3, SSM_NLDAS, P, R, ET, case, TR_argument, GN_std, input_FP, file_names, dth, p_threshold, event_opt)[0]
            
            p, dsm, dsm_or, SM1, SM2, dt, r, asm, et, infilt, res = make_input(df, case)
            idata, valid, lam = make_idata(p, dsm, dsm_or, SM1, SM2, dt, r, asm, et, infilt, res, case, method, event_opt) 
            save_idata(idata, valid, lam, case, method, TR, GN_std, save_dir, cell_id)
            return idata, valid, lam, df
        
    else:
        TR_argument = [TR, sub_opt, div_opt, sample_rate_opt] #TR, consider subsample (same TR), diversity, sample_rate
        df = make_df(SSM_SMAPL3, SSM_NLDAS, P, R, ET, case, TR_argument, GN_std, input_FP, file_names, dth, p_threshold, event_opt)[0]
        p, dsm, dsm_or, SM1, SM2, dt, r, asm, et, infilt, res = make_input(df, case)
        idata, valid, lam = make_idata(p, dsm, dsm_or, SM1, SM2, dt, r, asm, et, infilt, res, case, method, event_opt)
        return idata, valid, lam, df

### idata-related codes
def make_idata_list(save_dir):
    f = []
    for file in os.listdir(save_dir):
        if os.path.isfile(os.path.join(save_dir, file)) and file[0].isdigit():
            f.append(file)
    return f

def extract_posterior(save_dir, idata_file_name, var_name):
    #print(save_dir+idata_file_name)
    
    try:
        data = []
        with open(save_dir+idata_file_name, "rb") as f:
            while True:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break

        valid_point = data[0]
        cell_id = data[1]
        idata = data[2]

        if valid_point == 1:
            if isinstance(idata, list):
                var = idata[0][var_name]
            else:
                var = idata.posterior[var_name]
        else:
            print(str(cell_id), ' is not a valid point')
            var = [np.nan]
    except:
        #print(idata_file_name)
        print(idata_file_name, ' something is wrong with idata')
        var = [np.nan]
        cell_id = np.nan
        valid_point = 0
        idata = np.nan
    
    return var, cell_id, valid_point, idata

def calculate_hdi(idata, hdi_prob=0.94, hdi_type='lower'):
    sorted_data = np.sort(idata)
    n_samples = len(sorted_data)
    n_intervals = int(hdi_prob * n_samples)

    if hdi_type == 'lower':
        hdi_indices = np.arange(n_intervals)
    elif hdi_type == 'higher':
        hdi_indices = np.arange(n_samples - n_intervals, n_samples)
    else:
        raise ValueError("Invalid hdi_type. Choose 'lower' or 'higher'.")

    hdi_values = sorted_data[hdi_indices]
    return hdi_values
    
def extract_idata(idata_save_dir, lat, opt='median'):
    
    logging.getLogger("arviz").setLevel(logging.ERROR)
    idata_list = make_idata_list(idata_save_dir)
    Z=[0]*len(idata_list)
    cell_id=[0]*len(idata_list)
    Z_valid=[0]*len(idata_list)
    idata=[0]*len(idata_list)
    
    for i in range(len(idata_list)):
        Z[i],cell_id[i],Z_valid[i], idata[i] = extract_posterior(idata_save_dir, idata_list[i], 'Z')
 
    if Z_valid[i] == 1:
        cell_id = np.array(cell_id, dtype=np.int32)
        Z_temp=[0]*len(idata_list)
        for i in range(len(idata_list)):
            if Z_valid[i] == 1:
                z = Z[i]
                
                if not isinstance(z, np.ndarray):
                    z = z.values

                    if opt == 'median':
                        Z_temp[i] = np.median(z)
                    elif opt == 'hdi_low':
                        Z_samples = az.hdi(idata[i].posterior['Z'], hdi_prob=0.94)
                        Z_temp[i] = Z_samples['Z'].sel(hdi='lower').values
    
                    elif opt == 'hdi_high':
                        Z_samples = az.hdi(idata[i].posterior['Z'], hdi_prob=0.94)
                        Z_temp[i] = Z_samples['Z'].sel(hdi='higher').values
                    elif opt == 'std':
                        Z_temp[i] = np.std(z)
                else:
                    if opt == 'median':
                        Z_temp[i] = np.median(z)
                    elif opt == 'hdi_low':
                        Z_temp[i] = calculate_hdi(z, hdi_prob=0.94, hdi_type='lower')
                    elif opt == 'hdi_high':
                        Z_temp[i] = calculate_hdi(z, hdi_prob=0.94, hdi_type='higher')
                    elif opt == 'std':
                        Z_temp[i] = np.std(z)
            else:
                Z_temp[i] = np.nan 

        cell_id_python = Midx2Pidx(cell_id, lat.shape)
        Z_map = lat.copy()
        Z_map[:] = np.nan
        
    else:
        Z_temp[i] = np.nan
    Z_map.flat[cell_id_python] = Z_temp
    
    return Z_map

def delete_bad_cell_id(directory, bad_cell_id):
    for root, dirs, files in os.walk(directory):
        for file in files:
            for cell_id in bad_cell_id:
                if file.startswith(str(cell_id)):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    
def check_idata(idata_save_dir):
    idata_list = make_idata_list(idata_save_dir)
    bad_cell_id = []
    
    for i in range(len(idata_list)):

        try:
            data = []
            with open(idata_save_dir+idata_list[i], "rb") as f:
                while True:
                    try:
                        data.append(pickle.load(f))
                    except EOFError:
                        break
        except:
            bad_cell_id.append(idata_list[i].split("_")[0])
    bad_cell_id = list(map(int, bad_cell_id))
    
    return bad_cell_id

def Midx2Pidx(matlab_indices, matlab_shape):
    matlab_indices = np.array(matlab_indices) - 1  # Account for 1-based indexing
    row_indices, col_indices = np.unravel_index(matlab_indices, matlab_shape, order='F') # 'F' for column-major (Fortran) order
    python_indices = np.ravel_multi_index((row_indices, col_indices), matlab_shape, order='C') # 'C' for row-major (C) order
    return python_indices

### idata mapping code
def map_idata(lon, lat, value, title='idata', save_path=None, max_val=1, min_val=0, extent=[-120, -74, 22.5, 50], cmap = 'jet_r', cbar_title=r"Median $\Delta$Z[mm]"):
    fig = plt.figure(figsize=(10, 8))

    # Use the Albers Equal Area Conic projection
    prj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5, standard_parallels=(29.5, 45.5))
    ax = plt.axes(projection=prj)

    ax.coastlines(linewidth=0.5)
    ax.set_facecolor('gray')
    ax.set_extent(extent)

    # Set ocean color
    ocean_color = 'lightblue'
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color=ocean_color)

    ocean = cfeature.OCEAN
    ocean_data = ocean.intersecting_geometries(ax.get_extent())
    ocean_shape = cfeature.ShapelyFeature(ocean_data, ccrs.PlateCarree())

    ax.add_feature(ocean_shape, facecolor=ocean_color, edgecolor='black', linewidth=0.5)

    ax.set_facecolor('gray')
    ax.set_extent(extent)
    # Add US border
    us_border = cfeature.BORDERS
    ax.add_feature(us_border, edgecolor='black', linewidth=1)

    # Add state borders
    state_borders = cfeature.STATES.with_scale('50m')
    ax.add_feature(state_borders, edgecolor='black', linewidth=0.5)

    # Add river and lake colors
    ax.add_feature(cfeature.RIVERS, facecolor='lightblue', edgecolor='lightblue')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='lightblue')

    # Set the color of NaN values to gray
    cmap.set_bad(color='gray')

    # Plot the data
    plot = ax.pcolormesh(lon, lat, value, cmap=cmap, transform=ccrs.PlateCarree(), vmin=min_val, vmax=max_val)

    #plt.title(title, fontsize=16)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize': 16}
    gl.ylabel_style = {'fontsize': 16}

    cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.5, pad=0.02, aspect=30, location='top')
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(cbar_title, fontsize=20, labelpad=15)
    #cbar.set_ticks([0, max_val])

    # Save the plot as a PNG file
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, title.replace(' ', '_') + '.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

def plot_SM_P_for_event(df, SM, P, JDATES, ti=0, target_time=False, interval=10, offset=100):
    
    if len(df)==0:
        print('Not enough data to plot')
    else:
        if target_time:
            target_time = pd.to_datetime(target_time)
            ti = (df['t1'] - target_time).abs().idxmin()
        start_idx = df.start_idx
        end_idx   = df.end_idx
        P_start_idx = df.P_start_idx
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, dpi=150)

        ax1.plot([JDATES[start_idx[ti]], JDATES[end_idx[ti]]], [SM[start_idx[ti]], SM[end_idx[ti]]], '-rx')
        ax1.plot(JDATES[P_start_idx[ti]-offset:end_idx[ti]+1+offset], SM[P_start_idx[ti]-offset:end_idx[ti]+1+offset], '-k')
        ax1.set_ylabel('SM')

        ax2.bar(JDATES[P_start_idx[ti]-offset:end_idx[ti]+1+offset], P[P_start_idx[ti]-offset:end_idx[ti]+1+offset], width=0.02)
        ax2.set_xlabel('Dates')
        ax2.set_ylabel('P')

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval))  # Set the interval for tick marks
        plt.xticks(rotation=0)  # Rotate the x-axis labels

        plt.tight_layout()  # Adjust the layout to prevent labels from being cut off
        plt.show()

def plot_dSM_sumP(col, dSM, sumP, TR_argument, event_opt, ax=None):
    dt = TR_argument[0]
    
    if ax is None:
        fig, ax = plt.subplots(dpi=100)

    ax.scatter(dSM, sumP)
    ax.set_xlabel(r'$\Delta$SM', fontsize=20)
    ax.set_ylabel(r'$\sum P$', fontsize=20)
    ax.grid()
    
    # Set the title based on event_opt and dt
    if event_opt == 'P_dry':
        event = 'Dry-down'
    elif event_opt == 'P_wet':
        event = 'Wet-up'
    else:
        event = event_opt

    if isinstance(dt, str):
        title = "Location #{} \n".format(col) + r'$\Delta$t = {}-hr / {}'.format(dt, event)
        
    else:
        title = "Location #{} \n".format(col) + r'$\Delta$t = {:.0f}-hr / {}'.format(dt, event)
        
    ax.set_title(title, fontsize=20)

    # Fit a first-order regression line
    coeffs = np.polyfit(dSM, sumP, 1)
    x = np.linspace(min(dSM), max(dSM), 100)
    y = np.polyval(coeffs, x)
    ax.plot(x, y, 'r-', label=f'$\sum $P = {coeffs[0]:.2f} $\Delta$SM + {coeffs[1]:.2f}')
    ax.legend(prop=font_manager.FontProperties(size=14))

    if ax is None:  # Only call plt.show() if ax is None
        plt.show()
        
def plot_P_event_wetup_drydown(pi, SSM_NLDAS, P, R, ET, start_indices, end_indices, P_event_start_indices, P_event_end_indices, start_indices_or_SM, end_indices_or_SM, offset=30, maxnlocator=5):
    
    if len(start_indices)==0 or pi >= (len(start_indices)):
        print('Not enough data to plot')
    else:
        
        JDATES = pd.date_range(start='2015-01-01 00:30:00', end='2021-12-31 23:30:00', freq='1h').values.reshape(-1,)
        p_s_idx  = P_event_start_indices[pi]
        p_e_idx  = P_event_end_indices[pi]
        sm_s_idx = start_indices[pi]
        sm_e_idx = end_indices[pi]
        sm_or_s_idx = start_indices_or_SM[pi]
        sm_or_e_idx = end_indices_or_SM[pi]
        
        ss_idx = min([p_s_idx, sm_s_idx])
        ee_idx = max([p_e_idx, sm_e_idx])

        # Plot the precipitation data
        fig, ax1 = plt.subplots(dpi=100)
        ax1.plot(JDATES[ss_idx-offset:ee_idx+1+offset], P[ss_idx-offset:ee_idx+1+offset], 'b', label='P')
        ax1.plot(JDATES[ss_idx-offset:ee_idx+1+offset], R[ss_idx-offset:ee_idx+1+offset], 'c', label='R')
        ax1.plot(JDATES[ss_idx-offset:ee_idx+1+offset], ET[ss_idx-offset:ee_idx+1+offset], 'r', label='ET')
        
        ax1.bar(JDATES[p_s_idx:p_e_idx+1], P[p_s_idx:p_e_idx+1], color='b', alpha=0.5, width=0.05, edgecolor='w' )
        ax1.set_xlabel('Date')
        #ax1.set_ylim([0, 15])
        ax1.invert_yaxis()
        ax1.set_ylabel('Precipitation, R, ET (mm)')

        # Create a second y-axis for the soil moisture data
        ax2 = ax1.twinx()
        #ax2.set_ylim([0.27, 0.36])
        ax2.plot(JDATES[ss_idx-offset:ee_idx+1+offset], SSM_NLDAS[ss_idx-offset:ee_idx+1+offset], '-k', label='SM')
        ax2.plot([JDATES[sm_s_idx], JDATES[sm_e_idx]], [SSM_NLDAS[sm_s_idx], SSM_NLDAS[sm_e_idx]], 'g', linewidth=3, alpha=1, label='TR-hr dSM')
        ax2.plot([JDATES[sm_or_s_idx], JDATES[sm_or_e_idx]], [SSM_NLDAS[sm_or_s_idx], SSM_NLDAS[sm_or_e_idx]], '-.m', linewidth=3, alpha=1, label='True dSM')
        
        ax2.set_ylabel('Soil Moisture (m\u00B3/m\u00B3)')

        # Set the x-axis tick locator and formatter
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(maxnlocator))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # Display legends for all plotted variables
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2)
        plt.show()

def plot_P_event_both_wetup_drydown(pi, SSM_NLDAS, P, R, ET, P_start_indices, P_end_indices, wet_event_indices, dry_event_indices, wet_dry_v_event_indices, offset=30, maxnlocator=5):
        
    if len(wet_dry_v_event_indices)==0 or pi >= (len(wet_dry_v_event_indices)):
        print('Not enough data to plot')
    else:
        JDATES = pd.date_range(start='2015-01-01 00:30:00', end='2021-12-31 23:30:00', freq='1h').values.reshape(-1,)
        pi = wet_dry_v_event_indices[pi]
        s_idx = P_start_indices[pi]
        e_idx = P_end_indices[pi]
        w_s_idx = wet_event_indices[pi][0]
        w_e_idx = wet_event_indices[pi][-1]
        d_s_idx = dry_event_indices[pi][0]
        d_e_idx = dry_event_indices[pi][-1]

        ss_idx = min([s_idx, w_s_idx, d_s_idx])
        ee_idx = max([e_idx, w_e_idx, d_e_idx])

        # Plot the precipitation data
        fig, ax1 = plt.subplots(dpi=100)
        ax1.plot(JDATES[ss_idx-offset:ee_idx+1+offset], P[ss_idx-offset:ee_idx+1+offset], label='P')
        ax1.plot(JDATES[ss_idx-offset:ee_idx+1+offset], R[ss_idx-offset:ee_idx+1+offset], label='R')
        ax1.plot(JDATES[ss_idx-offset:ee_idx+1+offset], ET[ss_idx-offset:ee_idx+1+offset], label='ET')
        ax1.bar(JDATES[ss_idx:ee_idx+1], P[ss_idx:ee_idx+1], color='b', alpha=1, width=0.05)
        ax1.set_xlabel('Date')
        ax1.set_ylim([0, 4])
        ax1.invert_yaxis()
        ax1.set_ylabel('Precipitation (mm)')

        # Create a second y-axis for the soil moisture data
        ax2 = ax1.twinx()
        ax2.plot(JDATES[ss_idx-offset:ee_idx+1+offset], SSM_NLDAS[ss_idx-offset:ee_idx+1+offset], '-.g', 'SM')
        ax2.plot([JDATES[w_s_idx], JDATES[w_e_idx]], [SSM_NLDAS[w_s_idx], SSM_NLDAS[w_e_idx]], 'k', linewidth=3)
        ax2.plot([JDATES[d_s_idx], JDATES[d_e_idx]], [SSM_NLDAS[d_s_idx], SSM_NLDAS[d_e_idx]], 'r', linewidth=3)

        ax2.set_ylabel('Soil Moisture (m\u00B3/m\u00B3)')

        # Set the x-axis tick locator and formatter
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(maxnlocator))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # Display legends for all plotted variables
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2)
        plt.show()