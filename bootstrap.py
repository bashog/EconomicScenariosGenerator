import pandas as pd
import datetime
import random
from matplotlib import pyplot as plt
from esg import ESG
import time

class Bootstrap(ESG):
  ''' 
  Class to generate bootstrap samples 
  '''
  def __init__(self, data:pd.DataFrame, test_date: datetime.datetime, scenarios:int):
    ''' 
    Initialise the Bootstrap class with the ESG abstract class
    '''
    super().__init__(data, test_date, scenarios) 
  
  def pre_processing(self):
    ''' 
    Preprocess the data but do nothing for the bootstrap 
    '''
    pass
  
  def train(self):
    ''' 
    Generate scenarios based on the train set between the start date and the test date 
    '''
    self.output = []
    start = time.time()
    for i in range(self.scenarios):
      samples = self.data_train.sample(n=len(self.data_train), replace=True)
      samples.columns = [str(col) + f'_{i}' for col in self.columns]
      samples.index = self.data_train.index
      self.output.append(samples)
    end = time.time()
    self.time_train = end - start

  def generate(self):
    ''' 
    Generate scenarios based on the test set between the test date and the end date 
    and concatenate the train and test set 
    '''
    self.generated_samples = []
    start = time.time()
    for i in range(self.scenarios):
      samples = self.data_train.sample(n=len(self.data_test), replace=True)
      samples.columns = [str(col) + f'_{i}' for col in self.columns]
      samples.index = self.data_test.index
      samples = pd.concat([self.output[i], samples], axis=0)
      self.generated_samples.append(samples)   
    end = time.time()
    self.time_generate = end - start