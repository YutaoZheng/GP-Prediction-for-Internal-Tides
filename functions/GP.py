import xarray as xr
import numpy as np
from gptide import GPtideScipy
import numpy as np
from . import Processing
from . import Cov
from speccy import sick_tricks as gary
from speccy import utils as ut
import string
import pandas as pd
from sklearn.metrics import mean_absolute_error


def Sampling(covparams,sample_length,sample_freq,
             mean_func,mean_params,cov_model,sample_number=1,noise=0):
    xd = np.arange(0,sample_length*sample_freq,sample_freq)[:,None]
    print('the sampling duration is {} days'.format(sample_length*sample_freq))
    GP  = GPtideScipy(xd, xd, noise, cov_model, covparams,mean_func = mean_func,mean_params = mean_params)
    GP_samples = GP.prior(samples=sample_number)
    GP_samples_array = xr.DataArray(GP_samples.T,
                                  dims=('sample','time'),
                                  coords={"sample": np.arange(sample_number), "time": xd.flatten()}) #time in second
    GP_samples_dataset = xr.Dataset({"Amplitude": GP_samples_array})
    
    return GP_samples_dataset

def Extract_nonoverlapping_samples(data, sample_duration_days=28, 
                                   max_overlap_fraction=0.1, max_nan_fraction=0.1,
                                   max_samples=10, seed=None):
    if seed:
        np.random.seed(seed)
    # Convert time to floating seconds since t0
    time_values = data.time.values
    if np.issubdtype(time_values.dtype, np.datetime64):
        time_seconds = (time_values - time_values[0]) / np.timedelta64(1, 's')
    elif np.issubdtype(time_values.dtype, np.floating):
        time_values = time_values/1e9
        data = data.assign_coords(time=time_values)
        time_seconds = (time_values - time_values[0])
    else:
        raise TypeError(f"Unsupported time dtype: {time_values.dtype}")
    t0_abs = data.time.values[0]  # absolute start time (datetime64 or float)
    # Define duration in seconds
    duration_sec = sample_duration_days * 86400
    max_overlap_sec = duration_sec * max_overlap_fraction
    print(f"Max allowed overlap: {max_overlap_sec/86400:.1f} days ({max_overlap_sec} s)")
    start_sec = time_seconds[0]
    end_sec = time_seconds[-1] - duration_sec
    candidate_times = np.arange(start_sec, end_sec, 86400)  # 1-day steps
    np.random.shuffle(candidate_times)
    selected_samples = []
    selected_windows = []
    # Estimate sampling frequency (in seconds)
    time_diff = np.diff(time_seconds[:2])[0]
    samples_per_day = int(86400 / time_diff)
    expected_len = sample_duration_days * samples_per_day
    print(f"Expected points per sample: {expected_len} (based on {samples_per_day} samples/day)")
    print(f"Attempting to extract up to {max_samples} samples...")

    for base_t0_sec in candidate_times:
        if len(selected_samples) >= max_samples:
            break
        offset = np.random.uniform(0, 86400)
        t0_sec = base_t0_sec + offset
        t1_sec = t0_sec + duration_sec
        # Convert to absolute time units
        t0_abs_sel = t0_abs + np.timedelta64(int(t0_sec), 's') if np.issubdtype(data.time.dtype, np.datetime64) else t0_abs + t0_sec
        t1_abs_sel = t0_abs + np.timedelta64(int(t1_sec), 's') if np.issubdtype(data.time.dtype, np.datetime64) else t0_abs + t1_sec
        sample = data.sel(time=slice(t0_abs_sel, t1_abs_sel))
        print(f"\nChecking window: {t0_abs_sel} to {t1_abs_sel} | Points: {len(sample.time)}")
        if len(sample.time) < 2:
            print("  ❌ Too few data points.")
            continue
        tolerance = 10
        if not (expected_len - tolerance <= len(sample.time) <= expected_len + tolerance):
            print(f"  ❌ Sample length {len(sample.time)} out of allowed range [{expected_len - tolerance}, {expected_len + tolerance}].")
            continue
        # NaN check
        nan_fraction = np.isnan(sample).mean().item()
        print(f"  → NaN fraction: {nan_fraction:.2%}")
        if nan_fraction > max_nan_fraction:
            print("  ❌ Too many NaNs — rejected.")
            continue
        # Overlap check (in float seconds)
        too_much_overlap = False
        for prev_start_sec, prev_end_sec in selected_windows:
            overlap = min(t1_sec, prev_end_sec) - max(t0_sec, prev_start_sec)
            if overlap > max_overlap_sec:
                too_much_overlap = True
                break
        if too_much_overlap:
            print("  ❌ Overlaps too much with previous window — rejected.")
            continue
        else:
            print("  ✓ Overlap check passed.")
        # Sampling interval check
        time_diffs = np.diff(sample.time.values).astype('timedelta64[s]').astype(int)
        if not np.all(time_diffs == time_diffs[0]):
            print("  ❌ Inconsistent sampling interval — rejected.")
            continue
        else:
            print(f"  ✓ Constant sampling frequency: {time_diffs[0]} s")
        # All checks passed
        print("  ✅ Sample accepted.")
        selected_samples.append(sample)
        selected_windows.append((t0_sec, t1_sec))
    print(f"\nFinished. {len(selected_samples)} sample(s) extracted.")
    return selected_samples


def Predict_set_up(A_obs,obs_len,predict_len,freq_scale=False):
    #select a period of obs
    if np.issubdtype(A_obs.time.dtype, np.datetime64):
        obs_freq = (A_obs.time[1]-A_obs.time[0]).values.astype('float')/1e9/86400 #in days
    elif np.issubdtype(A_obs.time.dtype, np.floating):
        obs_freq = (A_obs.time[1]-A_obs.time[0]).values/86400 #in days
    else:
        raise TypeError(f"Unsupported time dtype: {A_obs.time.dtype}")
    x_obs = np.arange(0,obs_len*obs_freq,obs_freq)
    y_obs = A_obs.isel(time=slice(0, obs_len))
    print('the observation period is {} days'.format(obs_len*obs_freq))
    #set up prediction length
    if not freq_scale:
        predict_freq = obs_freq
        print('prediction interval is {} s'.format(predict_freq*86400))
    else:
        predict_freq = obs_freq*freq_scale
        print('prediction interval is {} s'.format(predict_freq*86400))
    x_predict = np.arange(0,predict_len*predict_freq,predict_freq)
    #set up the true 
    y_true = A_obs[obs_len:int(predict_len*predict_freq/obs_freq)][::int(predict_freq/obs_freq)]
    x_true = x_predict[obs_len:]
    return x_obs,y_obs,x_predict,x_true,y_true


def Predicting(A_obs,obs_len,predict_len,
               covparams,cov_model,
               mean_func,mean_params,
               sample_number=2, noise=0, error_threshold=40,
               freq_scale=False):
    
    x_obs,y_obs,x_predict,x_true,y_true = Predict_set_up(A_obs,obs_len,predict_len,
                                                            freq_scale = freq_scale)
    if len(x_true) == len(x_predict[x_predict > x_obs[-1]]): #check the length of the prediction
        print('the prediction is for {} days'.format(x_predict[-1]-x_obs[-1]))
        # set up GP
        OI = GPtideScipy(x_obs[:,None], x_predict[:,None], noise, 
                        cov_model, covparams,mean_func = mean_func,mean_params = mean_params)
        print('the prediction is the {} samples of the posterior + mean'.format(sample_number-1))
        y_predict_mean = OI(y_obs.values[:,None])[:, 0]
        y_predictions = OI.conditional(y_obs.values[:,None],samples=sample_number-1)
        # # Compute prediction error
        # mae = mean_absolute_error(y_true, y_predict_mean[len(y_obs):])
        # Compute errors
        abs_errors = np.abs(y_true - y_predict_mean[len(y_obs):])
        mae = abs_errors.mean()
        max_error = abs_errors.max()
        if mae > error_threshold or max_error > error_threshold:
            GP_predict_dataset = None
            print(f"Bad prediction: MAE = {mae:.2f}, Max Error = {max_error:.2f} (threshold {error_threshold})")
        else:
            # Combine mean + samples: sample[0] = mean, others = samples
            all_predictions = np.concatenate([y_predict_mean[:, None], y_predictions], axis=1).T  # shape (sample_number, n_predict)
            # put the prediction into a dataset
            GP_predict_array = xr.DataArray(all_predictions,
                                    dims=('sample','time'),
                                    coords={"sample": np.arange(sample_number), "time": x_predict},
                                    name="Amplitude") #time in second
            GP_predict_dataset = xr.Dataset({"Amplitude": GP_predict_array})
            GP_predict_dataset.attrs['obs_start'] = A_obs[0].time.values.astype('datetime64[D]')
            print('Prediction is done')
    else:
        print('The prediction length is not equal to the true length, stop the prediction')
        GP_predict_dataset = None
        
    return x_obs,y_obs,x_true,y_true,GP_predict_dataset

def M1P1(x,xpr,params):
    D2_freq = 1.93 #cpd
    M2_freq = 2 #cpd
    D2_freq = (D2_freq+M2_freq)/2
    
    η_matern1 = params[0]
    α_matern1 = params[1]
    eta_D2    = params[2]
    tau_D2    = params[3]
    gamma_D2  = params[4]
    
    dx    = np.sqrt((x-xpr)*(x-xpr))
    #background energy continuum  
    matern1 = Cov.Matern(dx, (η_matern1,α_matern1),lmbda=3,sigma=1e-6)   
    #peak
    peak2   = Cov.LR_2(dx, (eta_D2,tau_D2,gamma_D2),l_cos=D2_freq) 
    COV = matern1 + peak2 #+ noise
    return COV


def M1P2_2(x,xpr,params):
    S2_freq = 1.93 #cpd
    M2_freq = 2 #cpd

    η_matern1 = params[0]
    α_matern1 = params[1]
    eta_S2    = params[2]
    tau_S2    = params[3]
    gamma_S2  = params[4]
    eta_M2    = params[5]
    tau_M2    = params[6]
    gamma_M2  = params[7]
    
    dx    = np.sqrt((x-xpr)*(x-xpr))
    matern1 = Cov.Matern(dx, (η_matern1,α_matern1),lmbda=3,sigma=1e-6)              #background energy continuum  
    peak2   = Cov.LR_2(dx, (eta_S2,tau_S2,gamma_S2),l_cos=S2_freq) + \
              Cov.LR_2(dx, (eta_M2,tau_M2,gamma_M2),l_cos=M2_freq)
    COV = matern1 + peak2 #+ noise
    return COV




