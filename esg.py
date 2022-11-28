from abc import ABC, abstractmethod
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from utils import coumpound_quantiles

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
 

    @abstractmethod
    def pre_processing(self):
        ''' 
        Pre-processing of the data to implement in the inherited class 
        '''
        pass


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

    
    def correlation(self):
        ''' 
        Get the correlation matrix of bootstrap samples 
        The correlation matrix is computed by taking the average of the correlation matrix of each generated scenario 
        '''        
        if self.corr is None: # if the correlation matrix is not computed yet
            all_corr = []
            for sample in self.generated_samples: # for each generated scenario
                temp_samples = sample.copy()
                temp_samples.columns = self.columns
                all_corr.append(temp_samples.corr())
            self.corr = pd.concat(all_corr).groupby(level=0).mean() # average of the correlation matrix of each generated scenario
            self.corr = self.corr.loc[self.columns, self.columns] # reorder the columns
        sns.heatmap(self.corr, annot=True, cmap='crest')
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

        fig, ax = plt.subplots(self.ncols, 1, figsize=(15, 6*self.ncols))
        for i, col in enumerate(self.columns):
            sns.lineplot(x=self.index, y=self.data[col], ax=ax[i], color='blue', linewidth=1, label='Historical data')
            ax[i].set_title('Evolution of '+ col +' over time')
            ax[i].grid(True)

            quantiles_i = self.all_quantiles[i].rolling(windows).mean().bfill() # rolling mean of the quantiles

            # plot the quantiles
            #sns.lineplot(x=temp_index, y=quantiles_i[col+'_q50'], ax=ax[i], color='green', linewidth=1)
            sns.lineplot(x=self.index, y=quantiles_i[col+'_q10'], ax=ax[i], color='orange', linewidth=0.7,label='10% quantile')
            sns.lineplot(x=self.index, y=quantiles_i[col+'_q90'], ax=ax[i], color='orange', linewidth=0.7, label='90% quantile')
            sns.lineplot(x=self.index, y=quantiles_i[col+'_q2.5'], ax=ax[i], color='red', linewidth=0.7, label='2.5% quantile')
            sns.lineplot(x=self.index, y=quantiles_i[col+'_q97.5'], ax=ax[i], color='red', linewidth=0.7, label='97.5% quantile')

            # plot the historical quantile
            ax[i].axhline(self.data_train[col].quantile(0.10), color='orange', linestyle='dashed', linewidth=2, label='Historical 10% quantile')
            ax[i].axhline(self.data_train[col].quantile(0.90), color='orange', linestyle='dashed', linewidth=2, label='Historical 90% quantile')
            ax[i].axhline(self.data_train[col].quantile(0.025), color='red', linestyle='dashed', linewidth=2, label='Historical 2.5% quantile')
            ax[i].axhline(self.data_train[col].quantile(0.975), color='red', linestyle='dashed', linewidth=2, label='Historical 97.5% quantile')


            ax[i].legend(bbox_to_anchor = (1.22, 0.6), loc='center right')

            ax[i].axvline(x=pd.to_datetime(self.test_date, format='%Y-%m-%d'), color='k', linestyle='dashed', linewidth=2) # plot the test date line

            ax[i].set_xlim([pd.to_datetime(plot_from, format='%Y-%m-%d'), self.index[-1]]) # set the x-axis limits to reduce the size of the plot to focus on the last part of the data

        plt.show()
    
