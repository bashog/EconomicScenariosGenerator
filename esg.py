from abc import ABC, abstractmethod
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from utils import coumpound_quantiles, plot_quantiles_esg, plot_coumpound_quantiles_esg

class ESG(ABC):
    ''' 
    Abstract class for the Economic Scenario Generator 
    It is the blueprint for the inheritance of the different ESG classes
    
    The class must implement the following methods:
        - pre_processing : Preprocess the data
        - train : Train the model on the train set
        - generate : Generate the scenarios
        - quantiles : Get the quantiles of the generated data
        - correlation : Get the correlation matrix of the generated data
        - plot_returns : Plot the returns of the initial data and the quantiles of the generated data
        
    The class must have the following attributes:
        - data : pd.DataFrame 
            The data to be used for the ESG corresponding of the returns of the assets
        - index : pd.DatetimeIndex
            The index of the data
        - columns : pd.Index
            The columns of the data
        - ncols : int
            The number of columns of the data
        - scenarios : int
            The number of scenarios to be generated
        - test_date : str
            The date to create the train and test set
        - data_train : pd.DataFrame
            The train set
        - data_test : pd.DataFrame
            The test set
        - output : pd.DataFrame
            The output of the model after training
        - generated_samples : list
            The list of generated samples
        - all_quantiles : list
            The list of quantiles of the generated data for each asset
        - corr : pd.DataFrame
            The correlation matrix of the generated data
        '''

    def __init__(self, data:pd.DataFrame, test_date:str, scenarios:int):
        ''' 
        Initialize the ESG class

        Parameters:
        data: pd.DataFrame
            The data to be used for the ESG corresponding of the returns of the assets
        test_date: str
            The date to create the train and test set
        scenarios: int
            The number of scenarios to be generated
        '''
        self.name = self.__class__.__name__
        
        self.data = data
        self.index = data.index
        self.columns = data.columns
        self.ncols = len(self.columns)

        self.scenarios = scenarios

        self.test_date = test_date      
        self.data_train = self.data[self.data.index < self.test_date] # train set
        self.data_test = self.data[self.data.index >= self.test_date] # test set
        
        self.output = None
        self.generated_samples = None # list of generated samples
        self.all_quantiles = None # list of quantiles of the generated data for each asset
        self.corr = None # correlation matrix of the generated data
        
        self.time_train = None
        self.time_generate = None
 

    @abstractmethod
    def pre_processing(self):
        ''' 
        Pre-processing of the data to implement in the inherited class 
        '''
        pass
    
    def performance(self):
        ''' 
        Performance of the model during the training and the generation
        '''
        perf = pd.DataFrame(index=['train', 'generate'], columns=[self.name])
        perf.loc['train',] = [self.time_train]
        perf.loc['generate',] = [self.time_generate]
        return perf
        
    @abstractmethod
    def train(self):
        ''' 
        Train the model to implement in the inherited class 
        '''
        pass


    @abstractmethod
    def generate(self):
        ''' 
        Generate the scenarios to implement in the inherited class 
        '''
        pass
   
    
    def quantiles(self):
        ''' 
        Get the quantiles of the generated data 
        '''
        if self.all_quantiles is None:
            self.all_quantiles = [] # list of quantiles of the generated data for each asset
            for col in self.columns:
                samples_col = [sample.filter(regex=col) for i, sample in enumerate(self.generated_samples)] # get all the sample corresponding to the asset
                samples_col = pd.concat(samples_col, axis=1)
                quantiles_col = samples_col.quantile([0.025, 0.10, 0.5, 0.90, 0.975], axis=1).T # get the quantiles of the samples
                quantiles_col.columns = [col + '_q2.5', col + '_q10', col + '_q50', col + '_q90', col + '_q97.5'] # rename the columns
                self.all_quantiles.append(quantiles_col)
        print('Quantiles done')

    
    def correlation(self, corr_of='generated'):
        ''' 
        Get the correlation matrix of bootstrap samples 
        The correlation matrix is computed by taking the average of the correlation matrix of each generated scenario 
        '''        
        if corr_of == 'output':
            samples = self.output
            self.corr = samples.corr()
            self.corr = self.corr.loc[self.columns, self.columns] # reorder the columns
            title = 'Correlation matrix of the output'

        elif corr_of == 'generated':
            samples = self.generated_samples
            all_corr = []
            for sample in samples: # for each generated scenario
                temp_samples = sample.copy()
                temp_samples.columns = self.columns
                all_corr.append(temp_samples.corr())
                self.corr = pd.concat(all_corr).groupby(level=0).mean() # average of the correlation matrix of each generated scenario
                self.corr = self.corr.loc[self.columns, self.columns] # reorder the columns
                title = 'Correlation matrix of the generated data'

        sns.heatmap(self.corr, annot=True, cmap='crest')
        plt.title(title)
        plt.show()
        print('Correlation done')
        

    def plot_returns(self, plot_from:str, windows:int):
        ''' 
        Plot the returns of the generated data with the quantiles 0.025, 0.10, 0.5, 0.90, 0.975 
        The plot is done for each column of the data 
            
        Parameters:
        plot_from: str
            The date from which the plot is done
        windows: int
            Size of the windows of the rolling mean of the quantiles
        '''
        if self.all_quantiles is None: # if the quantiles are not computed yet
            self.quantiles()
            
        plot_quantiles_esg(self.data, self.data_train, self.all_quantiles, windows, self.test_date, plot_from)
    
    def plot_coumpound_returns(self, plot_from:str, prices:pd.DataFrame, method_returns:str):
        ''' 
        Plot the compound returns of the generated data with the quantiles 0.025, 0.10, 0.5, 0.90, 0.975 
        The plot is done for each column of the data 
            
        Parameters:
        plot_from: str
            The date from which the plot is done
        prices: pd.DataFrame
            Prices of the assets
        method_returns: str
            Method to compute the returns
        '''
        if self.all_quantiles is None: # if the quantiles are not computed yet
            self.quantiles()
        
        plot_coumpound_quantiles_esg(prices, self.all_quantiles, self.test_date, plot_from, method_returns)
    
