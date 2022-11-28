import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller

### About basic statistics ###



### About normality ###

def skewness(r:pd.Series):
    '''
    Computes the skewness of the supplied Series
    '''
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 3).mean()
    return exp / sigma_r ** 3


def kurtosis(r:pd.Series):
    '''
    Computes the kurtosis of the supplied Series or DataFrame
    '''
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 4).mean()
    return exp / sigma_r ** 4


def is_normal(r:pd.Series, level=0.01):
    '''
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = stats.jarque_bera(r)
        return p_value > level

### About stationnarity ###

def ADF(r:pd.Series):
    ''' 
    Calculate the Augmented Dickey-Fuller test of a time series of returns 
    '''
    return adfuller(r)[1]

def is_stationary(r:pd.Series, level=0.05):
    ''' 
    Check if a time series of returns is stationary 
    '''
    return ADF(r) < level

### About risk measure on returns ###

def get_quantiles(r:pd.Series, quantiles:list):
    ''' 
    Calculate the quantiles of a time series of returns 

    Parameters:
    r : pd.Series
        Time series of returns
    quantiles : list
        List of the quantiles we want to calculate
    '''
    return r.quantile(quantiles)


### Summary of statistics ###
def summary_statistics(r:pd.Series, symbol:str):
    ''' Calculate the summary statistics of a time series of returns '''
    index = ['Skewness', 'Kurtosis', 'Normality', 'Stationarity']
    summary = pd.DataFrame(columns=[symbol], index=index)
    summary.loc['Skewness',] = skewness(r)
    summary.loc['Kurtosis',] = kurtosis(r)
    summary.loc['Normality',] = is_normal(r)
    summary.loc['Stationarity',] = is_stationary(r)
    return summary



