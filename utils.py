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
    elif method_return == 'logarithmic':
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

## About geneeration of samples ##
def thermalisation_sampling(K, n_samples, prob_a, W, b, a):
    '''
    Thermalisation sampling method to generate new data
    It takes the precendent value and thermalise it for K steps
    '''
    sigmoid = lambda x: 1/(1+np.exp(-x))
    generated_samples_i = None
    v = np.random.binomial(1, prob_a)
    for _ in range(n_samples):
        for _ in range(K):
            h = sigmoid(np.dot(v, W) + b)
            h = np.random.binomial(1, h)
            v = sigmoid(np.dot(h, W.T) + a)
            v = np.random.binomial(1, v)
        generated_samples_i = np.concatenate((generated_samples_i, np.array([v])), axis=0) if generated_samples_i is not None else np.array([v])
    return generated_samples_i   


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

def plot_quantiles_esg(data:pd.DataFrame, data_train:pd.DataFrame, quantiles:list, windows:int, test_date:str, plot_from:str):
    ''' 
    Used to plot the quantiles returns from the generated datas
    
    Parameters:
    data: pd.DataFrame
        The hitorical returns of the assets
    data_train: pd.DataFrame
        The training data used to generate the historical quantiles to compare with the generated quantiles
    quantiles: list
        The list of quantiles for each asset from the generated data
    window: int
        The window for the rolling quantiles
    '''
    ncols = data.shape[1]
    index = data.index
    columns = data.columns


    fig, ax = plt.subplots(ncols, 1, figsize=(15, 6*ncols))
    for i, col in enumerate(columns):
        sns.lineplot(x=index, y=data[col], ax=ax[i], color='blue', linewidth=1, label='Historical data')
        ax[i].set_title('Evolution of '+ col +' over time')
        ax[i].grid(True)

        quantiles_i = quantiles[i].rolling(windows).mean().bfill() # rolling mean of the quantiles

        # plot the quantiles
        #sns.lineplot(x=temp_index, y=quantiles_i[col+'_q50'], ax=ax[i], color='green', linewidth=1)
        sns.lineplot(x=index, y=quantiles_i[col+'_q10'], ax=ax[i], color='orange', linewidth=0.7,label='10% quantile')
        sns.lineplot(x=index, y=quantiles_i[col+'_q90'], ax=ax[i], color='orange', linewidth=0.7, label='90% quantile')
        sns.lineplot(x=index, y=quantiles_i[col+'_q2.5'], ax=ax[i], color='red', linewidth=0.7, label='2.5% quantile')
        sns.lineplot(x=index, y=quantiles_i[col+'_q97.5'], ax=ax[i], color='red', linewidth=0.7, label='97.5% quantile')

        # plot the historical quantile
        ax[i].axhline(data_train[col].quantile(0.10), color='orange', linestyle='dashed', linewidth=2, label='Historical 10% quantile')
        ax[i].axhline(data_train[col].quantile(0.90), color='orange', linestyle='dashed', linewidth=2, label='Historical 90% quantile')
        ax[i].axhline(data_train[col].quantile(0.025), color='red', linestyle='dashed', linewidth=2, label='Historical 2.5% quantile')
        ax[i].axhline(data_train[col].quantile(0.975), color='red', linestyle='dashed', linewidth=2, label='Historical 97.5% quantile')


        ax[i].legend(bbox_to_anchor = (1.22, 0.6), loc='center right')

        ax[i].axvline(x=pd.to_datetime(test_date, format='%Y-%m-%d'), color='k', linestyle='dashed', linewidth=2) # plot the test date line

        ax[i].set_xlim([pd.to_datetime(plot_from, format='%Y-%m-%d'), index[-1]]) # set the x-axis limits to reduce the size of the plot to focus on the last part of the data

    plt.show()

def plot_coumpound_quantiles_esg(prices, all_quantiles, test_date, plot_from, method_return):
    ''' 
    Used to plot the coumpound quantiles returns from the generated datas 
    
    Parameters:
    prices: pd.DataFrame
        The hitorical prices of the assets
    all_quantiles: list
        The list of quantiles for each asset from the generated data
    test_date: str
        The test date
    plot_from: str
        The date from which to plot the data
    '''
    ncols = prices.shape[1]
    columns = prices.columns

    fig, ax = plt.subplots(ncols, 1, figsize=(15, 6*ncols))
    for i, col in enumerate(columns):
        # restrict the data from the date plot_from
        prices_i = prices[col][pd.to_datetime(plot_from, format='%Y-%m-%d'):]
        index_price_i = prices_i.index
        sns.lineplot(x=index_price_i, y=prices_i, ax=ax[i], color='blue', linewidth=1, label='Historical data')
        ax[i].set_title('Evolution of '+ col +' over time')
        ax[i].grid(True)

        # plot the coumpound quantile from the test date with the first value of the price for the coumpound
        # add the first price row to the quantiles and restrict to the test date
        quantiles_i = all_quantiles[i][pd.to_datetime(test_date, format='%Y-%m-%d'):]
        index_quantiles_i = quantiles_i.index
        #prices_i = prices_i[0]
        # calculate the cumulative product of the quantiles
        #quantiles_i = prices_i*(quantiles_i + 1).cumprod()
        price_test_date = prices[col][pd.to_datetime(test_date, format='%Y-%m-%d'):][0]
        if method_return == 'logarithmic':
            # cumulative log returens
            quantiles_i = price_test_date*np.exp(quantiles_i.cumsum())
        elif method_return == 'arithmetic':
            quantiles_i = price_test_date*(1+quantiles_i).cumprod()

        # plot the quantiles
        sns.lineplot(x=index_quantiles_i, y=quantiles_i[col+'_q10'], ax=ax[i], color='orange', linewidth=0.7,label='10% quantile')
        sns.lineplot(x=index_quantiles_i, y=quantiles_i[col+'_q90'], ax=ax[i], color='orange', linewidth=0.7, label='90% quantile')
        sns.lineplot(x=index_quantiles_i, y=quantiles_i[col+'_q2.5'], ax=ax[i], color='red', linewidth=0.7, label='2.5% quantile')
        sns.lineplot(x=index_quantiles_i, y=quantiles_i[col+'_q97.5'], ax=ax[i], color='red', linewidth=0.7, label='97.5% quantile')

        ax[i].legend(bbox_to_anchor = (1.22, 0.6), loc='center right')

        ax[i].axvline(x=pd.to_datetime(test_date, format='%Y-%m-%d'), color='k', linestyle='dashed', linewidth=2) # plot the test date line
       
    plt.show()   


        

