import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import math

import seaborn as sns
import matplotlib.pyplot as plt
import time
from tqdm import tqdm, trange
from utils import thermalisation_sampling, thermalisation_sampling_numba

from multiprocessing import Pool, cpu_count, Process

from esg import ESG

class RBM(ESG):
    def __init__(self, data:pd.DataFrame, test_date:str, scenarios:int):  
        '''
        Initialise the RBM class with the ESG abstract class
        input: encoded data
        W : weights
        a : bias for visible layer
        b : bias for hidden layer
        '''
        super().__init__(data, test_date, scenarios) 
        self.input = None 
        self.W = None 
        self.a = None
        self.b = None


        self.mse_a = None
        self.mse_b = None
        self.mse_W = None
        self.mse_abW = [[],[],[]]

    
    def encoding(self, array_real:np.array):
        '''
        Encode data into binary
        
        Parameters:
        array_real: numpy array of real data

        Returns:
        array_digits: numpy array of binary data
        '''
        array_digit = None
        for i in range(self.ncols):
            array_bin_i = list(map(lambda x: f'{int(65535 * (x - self.min[i]) / (self.max[i] - self.min[i])):016b}', array_real[:,i]))
            array_digit_i = np.array(list(map(lambda x: list(map(int, x)), array_bin_i)))
            array_digit = np.concatenate((array_digit, array_digit_i), axis=1) if array_digit is not None else array_digit_i
        return array_digit 

    
    def unencoding(self, array_digits:np.array):
        ''' 
        Decode binary data into real data
        
        Parameters:
        array_digits: numpy array of binary data'''   
        array_real = None
        for i in range(self.ncols):
            array_digits_i = array_digits[:,i*16:(i+1)*16]
            array_bin_i = list(map(lambda x: ''.join(map(str, x)), array_digits_i))
            array_real_i = list(map(lambda x: [int(x, 2) * (self.max[i] - self.min[i]) / 65535 + self.min[i]], array_bin_i))
            array_real = np.concatenate((array_real, np.array(array_real_i)), axis=1) if array_real is not None else np.array(array_real_i)
        return array_real


    def unpack_data(self, array:np.array, type_index:str):
        '''numpy to dataframe'''
        if type_index == 'train':
            index = self.data_train.index
        elif type_index == 'test':
            index = self.data_test.index
        elif type_index == 'all':
            index = self.data.index
        data = pd.DataFrame(array, columns=self.data.columns, index=index)
        return data


    def pre_processing(self):
        '''
        Pre-processing data train
        It takes the real data and encode it into binary by setting the min and max values used also for the encoding
        It initialises the weights and biases
        '''
        # convert data into numpy and encode them
        data_pack = self.data_train.to_numpy()
        self.min = np.amin(data_pack,axis=0)
        self.max = np.amax(data_pack,axis=0)
        self.n_visible = 16 * self.ncols
        self.n_hidden = int(7.5 * self.n_visible)
        self.input = self.encoding(data_pack)        

        # initialise weights and biases
        self.prob_a = np.mean(self.input, axis=0)
        self.a = np.log(self.prob_a / (1 - self.prob_a))
        self.b = np.zeros(self.n_hidden)
        self.W = np.random.normal(loc=0, scale=0.01, size=(self.n_visible, self.n_hidden))
        print('Pre-processing done')
    
    def performance(self):
        return super().performance()
    
    def qq_plot(self):
        '''QQ plot for each variable comparing the real data and the generated data'''
        fig, axes = plt.subplots(math.ceil(self.ncols/2), 2, figsize=(5*self.ncols, 5*self.ncols))
        for i, col in enumerate(self.columns):
            pp_array1 = sm.ProbPlot(self.data.iloc[:,i])
            pp_array2 = sm.ProbPlot(self.output.iloc[:,i])
            pp_array2.qqplot(line='45', other=pp_array1, xlabel='Output Quantiles'+col, ylabel='Real Quantiles'+col, ax=axes[i//2, i%2])
        plt.show()


    def mse(self, array1:np.array, array2:np.array):
        ''' 
        Calculate the mean square error between two arrays of real data
        Parameters:
            array1 (np.array)
            array2 (np.array)
        Returns:
            mean (float)
        '''
        return np.mean((array1 - array2) ** 2)
    

    def plot_mse(self):
        '''Plot the mse'''
        plt.figure(figsize=(15,6))
        plt.plot(pd.DataFrame(self.mse_abW[0]).rolling(10).mean(), label='mse of the visible units')
        plt.plot(pd.DataFrame(self.mse_abW[1]).rolling(10).mean(), label='mse of the hidden units')
        plt.plot(pd.DataFrame(self.mse_abW[2]).rolling(10).mean(), label='mse of the weights')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.yscale('log')
        plt.title('MSE of the visible bias, hidden bias and weights')
        plt.legend(bbox_to_anchor = (1.22, 0.6), loc='center right')
        plt.show()
       
    
    def train(self, epochs=1000, lr=0.01, batch_size=10, k_steps=1,):
        '''
        k contrastive divergence method with batch training
        The training is done on data train

        Parameters:
        epochs: number of epochs corresponding of the thermalisation time
        lr: learning rate
        batch_size: size of the batch used for training each epoch
        k_steps: number of steps for the contrastive divergence method
        
        At the end of the training, it takes the output of the RBM and decode it into real data
        '''
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        batchs = [self.input[i:i+batch_size] for i in range(0, len(self.input), batch_size)]
        n_batchs = len(batchs)

        t = trange(epochs, leave=True)
        start = time.time()
        for epoch in t: # thermalisation  

            batch = batchs[epoch % n_batchs]
            for elt in batch: # batch training
                v0 = v = elt # v0 is used to compute the gradient
                h0 = h = sigmoid(np.dot(v, self.W) + self.b) # h0 is used to compute the gradient
                h = np.random.binomial(1, h) # correspond of the first forward pass

                for _ in range(k_steps): # k steps of contrastive divergence
                    v = sigmoid(np.dot(h, self.W.T) + self.a)
                    v = np.random.binomial(1, v)
                    h = sigmoid(np.dot(v, self.W) + self.b)
                    h = np.random.binomial(1, h)
                
                #update weights and biases
                positive_grad = sigmoid(np.dot(v0, self.W) + self.b)
                negative_grad = sigmoid(np.dot(v, self.W) + self.b)
                
                self.W += lr * (np.outer(v0,positive_grad) - np.outer(v,negative_grad))
                self.a += lr * (v0 - v)
                self.b += lr * (h0 - h)

            #compute the mean square error of the bias of visible/hidden layers and on weights
            self.mse_a = self.mse(v0, v)
            self.mse_b = self.mse(h0, h)
            self.mse_W = self.mse(np.outer(v0,positive_grad), np.outer(v,negative_grad))

            self.mse_abW[0].append(self.mse_a)
            self.mse_abW[1].append(self.mse_b)
            self.mse_abW[2].append(self.mse_W)
            
        end = time.time()
        self.time_train = end - start

        self.output = None
        for elt in self.input: # take the output of the RBM and decode it into real data and plot the QQ plot to compare the real data and the output
            v = elt
            #forward pass
            h = sigmoid(np.dot(v, self.W) + self.b)
            h = np.random.binomial(1, h)
            #backward pass            
            v = sigmoid(np.dot(h, self.W.T) + self.a)
            v = np.random.binomial(1, v)
            self.output = np.concatenate((self.output, np.array([v])), axis=0) if self.output is not None else np.array([v])
        self.output = self.unencoding(self.output)
        self.output = self.unpack_data(self.output,'train')
        self.qq_plot()
        self.correlation(corr_of='output')

        print('Train done')

    
    def generate(self, K=1000, parallel=False, numba=False):
        '''
        Generate new data

        Parameters:
        method: 'simple' for simple generation by sampling with the historical distribution of data train
                'thermalisation' for generation new data different from the historical data train. It successively took the precendent value and thermalise it for K steps
        K: thermalisation time for the 'thermalisation' method to generate new data
        '''
        n_samples = self.data.shape[0]
        self.generated_samples = []
        
        if parallel:
            start = time.time()
            # parallelisation of the thermalisation sampling method to generate self.scenarios new data            
            with Pool(processes=cpu_count()) as pool:
                if numba:
                    self.generated_samples = pool.starmap(thermalisation_sampling_numba, [(K, n_samples, self.prob_a, self.W, self.b, self.a) for _ in range(self.scenarios)])
                else:
                    self.generated_samples = pool.starmap(thermalisation_sampling, [(K, n_samples, self.prob_a, self.W, self.b, self.a) for _ in range(self.scenarios)])

        else:
            start = time.time()
            for _ in trange(self.scenarios):
                if numba:
                    self.generated_samples.append(thermalisation_sampling_numba(K, n_samples, self.prob_a, self.W, self.b, self.a))
                else:
                    self.generated_samples.append(thermalisation_sampling(K, n_samples, self.prob_a, self.W, self.b, self.a))

        end = time.time()
        self.time_generate = end - start
        print('Time to generate the data: {}s'.format(end-start))               

               
        self.generated_samples = [self.unencoding(self.generated_samples[i]) for i in range(self.scenarios)] # unencoding
        
        self.generated_samples = [self.unpack_data(self.generated_samples[i], 'all') for i in range(self.scenarios)] # unpack data to get a dataframe        
        

    print('Generate data done')
    
    
class RBM_simple:
    def __init__(self, data:pd.DataFrame, n_visible:int, n_hidden:int):

        # about data
        self.data = data # real data
        self.array = None
        self.min, self.max = np.amin(data), np.amax(data)
        self.data_real = None # real data obtain after encoding and decoding
        self.mse_data = None # mse between data and data_real
        self.input = None # data in digits for the training

        # about network
        self.n_visible = n_visible # number of visible nodes
        self.n_hidden = n_hidden # number of hidden nodes
        self.W = np.random.normal(loc=0.0, scale=0.01, size=(n_visible, n_hidden)) # weight matrix
        self.a = None # bias of visible nodes
        self.b = None # bias of hidden nodes
        self.mse_errors = [] # list of mse error of training
        self.reconstruct_data = [] # list of reconstruct real data after training
        self.ouput_real = [] # list of output real data after training

    def encoding(self, array_real:np.array):
        '''
        Encode array of real data into array of digits
        Parameters:        
            array_real (np.array)
        Returns:
            array_digits (list)
        '''
        array_bin = list(map(lambda x: f'{int(65535 * (x - self.min) / (self.max - self.min)):016b}', array_real))
        array_digits = [list(map(int, list(bin))) for bin in array_bin]
        return np.array(array_digits)        
    
    def unencoding(self, array_digits:list):
        '''
        Decode array of digits into array of real data      
        Parameters:        
            array_digits (list)
        Returns:
            array_real (np.array)
        '''
        array_bin = [''.join(list(map(str,digit))) for digit in array_digits]
        array_int = list(map(lambda x: int(x,2), array_bin))
        array_real = list(map(lambda x: self.min + x * (self.max - self.min) / 65535, array_int))
        return np.array(array_real)

    def mse(self, array1:np.array, array2:np.array):
        ''' 
        Calculate the mean square error between two arrays of real data
        Parameters:
            array1 (np.array)
            array2 (np.array)
        Returns:
            mse (float)
        '''
        return np.mean((array1 - array2) ** 2) 
    
    def initialize_bias(self):
        '''Initialize the bias of visible and hidden nodes'''
        prob_a = np.mean(self.input, axis=0)
        self.a = np.log(prob_a / (1 - prob_a))
        self.b = np.zeros(self.n_hidden)
    
    def plot_data(self, data1:np.array, data2=None, title:str="", legend1:str="", legend2:str=""):
        '''Plot the data'''
        plt.figure(figsize=(10,5))
        sns.histplot(data1,bins=50,kde=True)    
        if data2 is not None:
            sns.histplot(data2,bins=50,kde=True)
        plt.title(title)
        plt.legend([legend1, legend2])
        plt.show() 
    
    def qqplot(self, array1:np.array, array2:np.array, title:str="", legend1:str="", legend2:str=""):
        '''Plot the qqplot of two arrays of real data'''
        plt.figure(figsize=(10,5))
        pp_array1 = sm.ProbPlot(array1)
        pp_array2 = sm.ProbPlot(array2)
        pp_array1.qqplot(line='45', other=pp_array2, label=legend2)
        plt.title(title)
        plt.legend([legend1, legend2])
        plt.show()

    def pre_processing(self):
        '''Pre-processing the data'''
        self.input = self.encoding(self.data)
        self.data_real = self.unencoding(self.input)
        self.mse_data = self.mse(self.data, self.data_real)
        print(f"min: {self.min}, max: {self.max}")
        print(f"mse: {self.mse_data}")
        self.plot_data(self.data_real)
        self.initialize_bias()
    
    def train(self, K, batch_size=10, k_steps=1, lr=0.01, verbose=True):
        '''k contrastive divergence method'''
        start_time = time.time()
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        '''Train the rbm'''
        t = trange(K, leave=True)
        batchs = [self.input[i:i+batch_size] for i in range(0, len(self.input), batch_size)]
        n_batchs = len(batchs)
        for epoch in t:
            output = []    
            batch = batchs[epoch % n_batchs]
            for elt in batch:
                v0 = v = elt
                #forward pass
                h0 = h = sigmoid(np.dot(v, self.W) + self.b)
                h = np.random.binomial(1, h)

                #backward pass
                for _ in range(k_steps):
                    v = sigmoid(np.dot(h, self.W.T) + self.a)
                    v = np.random.binomial(1, v)
                    h = sigmoid(np.dot(v, self.W) + self.b)
                    h = np.random.binomial(1, h)
                output.append(v)
                
                #update weights and biases
                positive_grad = sigmoid(np.dot(v0, self.W) + self.b)
                negative_grad = sigmoid(np.dot(v, self.W) + self.b)
                
                self.W += lr * (np.outer(v0,positive_grad) - np.outer(v,negative_grad))
                self.a += lr * (v0 - v)
                self.b += lr * (h0 - h)

            ouput_temp_real = self.unencoding(output)
            batch_real = self.unencoding(batch)
            mse_temp = self.mse(batch_real, ouput_temp_real)
            self.mse_errors.append(mse_temp)
            if verbose:
                t.set_description(f"mse: {mse_temp:.4f}")

        output_finale = []
        for elt in self.input:
            v = elt
            #forward pass
            h = sigmoid(np.dot(v, self.W) + self.b)
            h = np.random.binomial(1, h)
            #backward pass            
            v = sigmoid(np.dot(h, self.W.T) + self.a)
            v = np.random.binomial(1, v)
            output_finale.append(v)
        self.reconstruct_data = self.unencoding(output_finale)

        print(f"Training time: {time.time() - start_time}s")
        plt.figure(figsize=(10,5))
        sns.lineplot(self.mse_errors)
        plt.show()
        self.plot_data(self.reconstruct_data, self.data, title="Reconstruct data", legend1="Reconstruct", legend2="Real")
        self.qqplot(self.reconstruct_data, self.data, title="QQplot", legend1="Reconstruct", legend2="Real")
                   

        
    def generate_data(self, n_samples:int, mu:float, sigma:float):
        '''Generate new data'''
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        st = np.random.normal(loc=mu, scale=sigma, size=n_samples)
        st = st[ (st >= self.min) & (st <= self.max) ]
        st_encoded = self.encoding(st)
        st_gen = []
        for elt in st_encoded:
            h = sigmoid(np.dot(elt, self.W) + self.b)
            h = np.random.binomial(1, h)
            v = sigmoid(np.dot(h, self.W.T) + self.a)
            v = np.random.binomial(1, v)
            st_gen.append(v)
        st_gen = self.unencoding(st_gen)
        self.plot_data(st_gen)
        self.qqplot(self.data_real, st_gen, title="QQplot", legend1="Real", legend2="Generated")


        






    
