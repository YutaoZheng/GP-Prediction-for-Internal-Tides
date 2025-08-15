from unittest import result
import xarray as xr
import numpy as np


def Cal_SE(prediction,obs):
    """
    prediiction (xr.dataset)
    Astfalck2023 eq 7
    """
    # Compute root squared error
    SE_list = []
    for i in prediction:
        SE = np.power(i.values-obs.values,2)
        SE_list.append(SE)
    return SE_list

def Cal_DSS_1D(prediction,obs):
    """
    prediiction (xr.dataset) (1D)
    Astfalck2023 eq 8
    """
    m = len(prediction.sample)
    μ = prediction.mean(dim='sample').values
    Σ = ((prediction-μ)*(prediction-μ).T).sum(dim='sample')/(m-1)
    # Σ = prediction.std(dim='sample')**2
    diff = μ - obs.values
    result = np.log(abs(Σ)).values + diff.T*diff/Σ.values
    return result

def Cal_CRPS(prediction,obs):
    """
    prediiction (xr.dataset) (1D)
    Astfalck2023 eq 12
    """
    m = len(prediction.sample)
    #1st term - for accuracy and precision
    first_term = np.abs(prediction - obs.values).sum(dim='sample')/m
    #2ed term - only precision
    diff_list = []
    for i in prediction:
        diff = np.abs(i-prediction).sum(dim='sample')
        diff_list.append(diff.values)
    second_term = np.sum(diff_list,axis=0)/(2*m**2)
    return first_term - second_term    

#calulate avg skill score
def Cal_skill(true_y, obs_x, GP_predict_dataset,skill_type='SE'):
    if GP_predict_dataset is None:
        print("No valid prediction available.")
        return None
    # Select prediction after observation
    y_prediction = GP_predict_dataset['Amplitude'].where(GP_predict_dataset['time'] > obs_x[-1],drop=True)
    # Compute skill
    if skill_type == 'SE':
        skill = Cal_SE(y_prediction, true_y)
    elif skill_type == 'DSS':
         skill = Cal_DSS_1D(y_prediction, true_y)
    elif skill_type == 'CRPS':
         skill = Cal_CRPS(y_prediction, true_y)
    else:
        print(f"Unknown skill type: {skill_type}")
        skill =  None
    return skill