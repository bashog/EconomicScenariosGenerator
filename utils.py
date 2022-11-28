import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


### About returns ###
def get_returns(prices:pd.DataFrame, method_return:str, keep_extreme_value:bool):
    ''' Calculate the returns of a time series of prices '''
    if method_return == 'arithmetic':
        data_returns = prices.pct_change()
    elif method_return == 'log':
        prices = prices.astype(float)
        data_returns = np.log(prices/ prices.shift(1) )
    else:
        raise ValueError('type must be either arithmetic or log')
    
    if not keep_extreme_value:
        q99 = data_returns.quantile(0.99)
        q1 = data_returns.quantile(0.01)
        data_returns = data_returns[(data_returns < q99) & (data_returns > q1)]
    
    data_returns.dropna(inplace=True)
    return data_returns
    

def coumpound_quantiles(quantiles:pd.DataFrame,  prices:pd.DataFrame, method_return:str):
    ''' Calculate the confidence interval from the quantiles returns'''
    quantiles = quantiles.copy().reindex(prices.index)
    quantiles.fillna(0, inplace=True)
    
    if method_return == 'arithmetic':
        cum_quantiles = (quantiles + 1).cumprod()
        cum_quantiles = cum_quantiles * prices.iloc[0]
    elif method_return == 'log':
        # use the first value of prices for the cumulative quantiles
        cum_returns = prices.iloc[0] * np.exp(quantiles.cumsum())
    else:
        raise ValueError('type must be either arithmetic or log')

    return cum_quantiles

    


### To plot ###

def plot_prices(prices:pd.DataFrame):
    ''' Plot the prices of a time series of prices'''
    rows = prices.shape[1]
    index = prices.index
    # we use seaborn to plot the prices
    fig, ax = plt.subplots(rows, 1, figsize=(12, 5*rows))
    for i, col in enumerate(prices.columns):
        sns.lineplot(x=index, y=prices[col], ax=ax[i], color='blue', linewidth=0.7)
        ax[i].set_title('Evolution of '+ col +' over time')
        ax[i].grid(True)
    plt.show()
      

def plot_returns(returns:pd.DataFrame, with_quantile:bool=True):
    ''' Plot the returns of a time series of returns and the histogram of the returns'''
    rows = returns.shape[1]
    index = returns.index
    # we use seaborn to plot the prices
    fig, ax = plt.subplots(rows, 2, figsize=(15, 5*rows))
    for i, col in enumerate(returns.columns):
        sns.lineplot(x=index, y=returns[col], ax=ax[i, 0], color='blue', linewidth=0.5)
        ax[i, 0].set_title('Evolution of '+ col +' over time')
        ax[i, 0].grid(True)
        sns.histplot(returns[col], ax=ax[i, 1], color='blue')
        ax[i, 1].set_title('Histogram of '+ col)
        ax[i, 1].grid(True)
        if with_quantile:
            q99 = returns[col].quantile(0.99)
            q1 = returns[col].quantile(0.01)
            ax[i, 0].axhline(q99, color='red', linestyle='dashed')
            ax[i, 0].axhline(q1, color='red', linestyle='dashed')
            ax[i, 1].axvline(q99, color='red', linestyle='dashed')
            ax[i, 1].axvline(q1, color='red', linestyle='dashed')
    plt.show()


        

