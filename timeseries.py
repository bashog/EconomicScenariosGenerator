import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

from utils import get_returns, plot_prices, plot_returns
from statistics_tools import summary_statistics
from bootstrap import Bootstrap
from rbm import RBM


class TimeSeries:
  def __init__(self, data:pd.DataFrame, symbols:list):
    '''
    Initialization of the class
    data : pd.DataFrame
      Dataframe with the prices of the assets
    symbols : list
      List of the symbols of the assets we want to analyse
    n_cols : int
      Number of columns of the data we want to analyse
    method_return : str
      Method to calculate the returns 
      It can be 'arithmetic' or 'logarithmic'
    main_stat : pd.DataFrame
      Dataframe with the main statistics of the returns for each asset
    corr : pd.DataFrame
      Dataframe with the correlation matrix of the returns
    '''

    self.data = data[['Date',*symbols]].copy() # we select the columns we need
    self.symbols = symbols # name of rates
    self.n_cols = len(symbols)
    self.first_value = None

    self.method_return = None
    self.returns = None 
    self.main_stat = None
    self.corr = None

    self.returns_weekly = None
    self.returns_monthly = None
    self.returns_annualy = None

  def pre_processing(self, fill_method='fill', method_return='arithmetic', keep_extreme_value=False): 
    '''
    Basic preprocessing of the data
    fill_method : str
      Method to fill the missing values
      It can be 'fill' or 'drop'
    method_return : str
      Method to calculate the returns
      It can be 'arithmetic' or 'logarithmic'
    keep_extreme_value : bool
      If True, we keep the extreme values of the returns
    '''

    # we deal with the missing values
    if fill_method=='drop':
      self.data.dropna(inplace=True)
    elif fill_method=='fill':
      self.data.bfill(inplace=True)

    # we set the dates as index
    self.data.loc['Date'] = pd.to_datetime(self.data['Date'], format='%Y-%m-%d') # type of date is datetime
    self.data.set_index(keys=pd.DatetimeIndex(self.data['Date']), inplace=True) # index are the dates
    self.data = self.data.drop(columns=['Date']) # we drop the column Date
    self.data.dropna(inplace=True)
    
    # we calculate the returns
    self.method_return = method_return
    self.returns = get_returns(self.data, self.method_return, keep_extreme_value)

    # computation of all the returns (weekly, monthly and annualy)
    self.returns_weekly = self.returns.resample('W').sum()
    self.returns_monthly = self.returns.resample('M').sum()
    self.returns_annualy = self.returns.resample('Y').sum()


  def plot(self, type_plot:str, frequency='daily'):
    '''
    Plot the prices or the returns
    type_plot : str
      Type of plot we want to generate 
      It can be 'prices' or 'returns'
    type_return : str
      Type of returns we want to plot
      It can be 'daily', 'weekly', 'monthly' or 'annualy'
    '''
    if type_plot=='rates':
      plot_prices(self.data)

    elif type_plot=='returns':

      if frequency=='daily':
        plot_returns(self.returns)

      elif frequency=='monthly':
        plot_returns(self.returns_monthly)
      
      elif frequency=='weekly':
        plot_returns(self.returns_weekly)

      elif frequency=='annualy':
        plot_returns(self.returns_annualy)



  def statistics(self):
    '''Calculate the main statistics of the returns'''
    if self.main_stat is None:
      summary_stat = None 
      for symbol in self.symbols:
        summary_stat = pd.concat([summary_stat, summary_statistics(self.returns[symbol], symbol)], axis=1) if summary_stat is not None else summary_statistics(self.returns[symbol], symbol)
      self.main_stat = summary_stat
    return self.main_stat    

  def correlation(self):
    '''Calculate the correlation matrix'''
    if self.corr is None:
      self.corr = self.returns.corr()
    sns.heatmap(self.corr, annot=True, cmap='crest')
    plt.title('Historical correlation')
    plt.show()
  
  def bootstrap_esg(self, scenarios:int, test_date, plot_from:str, windows=5, frequency='daily'):
    '''Generate bootstrap samples'''
    if frequency == 'daily':
      self.bts = Bootstrap(self.returns, test_date, scenarios)
    elif frequency == 'weekly':
      self.bts = Bootstrap(self.returns_weekly, test_date, scenarios)
    elif frequency == 'monthly':
      self.bts = Bootstrap(self.returns_monthly, test_date, scenarios)
    elif frequency == 'annualy':
      self.bts = Bootstrap(self.returns_annualy, test_date, scenarios)
    self.bts.pre_processing()
    self.bts.train()
    self.bts.generate()
    self.bts.correlation()
    self.bts.plot_returns(plot_from, windows)
    print(self.bts.performance())
    

  def rbm_esg(self, scenarios:int, epochs:int, lr:float, K:int, numba: bool, test_date, plot_from:str, windows=10, frequency="daily", parallel=True):
    '''Generate RBM samples'''
    if frequency == 'daily':
      self.rbm = RBM(self.returns, test_date, scenarios)
    elif frequency == 'weekly':
      self.rbm = RBM(self.returns_weekly, test_date, scenarios)
    elif frequency == 'monthly':
      self.rbm = RBM(self.returns_monthly, test_date, scenarios)
    elif frequency == 'annualy':
      self.rbm = RBM(self.returns_annualy, test_date, scenarios)
    self.rbm.pre_processing()
    self.rbm.train(epochs, lr)
    self.rbm.plot_mse()
    self.rbm.generate(K, parallel, numba)
    self.rbm.correlation(corr_of='generated')
    self.rbm.plot_returns(plot_from, windows)
    self.rbm.plot_coumpound_returns(plot_from, self.data, self.method_return)
    print(self.rbm.performance())

    
  def performance(self):
    '''Return the performance of the RBM and the bootstrap'''
    perf_bts = self.bts.performance()
    perf_rbm = self.rbm.performance()
    return pd.concat([perf_bts, perf_rbm], axis=1)


