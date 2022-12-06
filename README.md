# Economic Scenario Generator

## Context

In the context of the "Counterparty Risk" project, which is related to the ability of the counterparty on the OTC markets to meet its commitments such as payment, delivery or repayment, it is essential to manage financial asset portfolios by taking into account different possible scenarios. This is where the 'ESG', an economic scenario generator, comes in to predict different risk factors over a period of several years.  

An ESG consists of a set of models (interest rates, equities, asset yields, credit spreads, inflation, etc.) allowing to randomly simulate possible evolution scenarios of the financial markets, which have an impact on the value and the performance of the assets. 

## Project motivations and purposes

After the financial crisis of 2008, states realized that economic risks could represent a threat to the community. As a result, stricter measures have been imposed on banks and insurance companies in terms of regulation and economic conduct. Among these measures we cite a better management of risks by modeling its factors. This project  aims to analyze and forecast the exchange rate as our main risk factor.  

Through an ESG we could simulate this variable by being consistent with its past evolutions, i.e. with the dynamics associated with it but also with the evolution of the dependency structure. Parametric or non-parametric methods like bootstrapping can be used, as well as generative machine learning models.  

## Structure of the code


## The models

### Time series class

### Economic scenario generator abstract class
- purpose of the class
- design of the class

### Bootstrapping
- explanation of the algorithm
- how to use it (snippet of code)
- results (photo etc)


### Restricted Boltzman Machine
#### RBM method
A Restricted Boltzmann Machine is a two-layer network with stochastic activation units. Restricted means that there is no connection between units whithin the same layer. The layer which is exposed to the training data set is called the visible layer. Inputs from the visible layer flow through the network (forward pass) to the hidden layer.

Each hidden unit then ”fires” randomly − its output is a Bernoulli random variable: ”1” is generated with probability p, which is equal to the sigmoid activation function value, and ”0” is generated with probability 1 − p. The outputs from the hidden layer then flow back (backward pass) to the visible layer, where they are aggregated and added to the visible layer biases.

The network learns the joint distribution of the configurations of visible and hidden activation units by trying to reconstruct the inputs from the training data set (visible unit values) by finding an optimal set of the network weights and biases

Here is a schema to visualize hidden and visible units :

<img width="404" alt="Capture d'écran_20221113_181203" src="https://user-images.githubusercontent.com/119663180/205937600-e31cb887-f52e-434c-bfd0-431dac77804e.png">

#### Code Structure
The RBM class gathers all the functions that allow to realize the model : 

- The __init__ function :  Initialise the RBM class with the ESG abstract class avec comme entrée les données encodées 

- The Encoding function: allows to transform the data into binary with the real data table as input 

- The Unencoding function: allows to decode the output of the binary algorithm into real data 

- The unpack_data function:  

- The pre_processing function : It takes the real data and encode it into binary by setting the min and max values used also for the encoding and It initialises the weights and biases 

- The qq_plot function : it draws the QQ plot for each variable to compare the real data and the generated data 

- The train function : it is the training function of the data with the k contrastive divergence method. For each batch among each of the epochs, weights and biases are updated and then for each epoch we compute the mean square error of the bias of visible/hidden layers and on weights. Then the generated output passes through the forward and backward passes and it is unencoded it the end.

- The generate function: allows to generate new data with the thermalization method which is based on the generation of new data. For each scenario, we take a random vector already trained on the previous weights and biases then we pass it K times (K represents the thermalization factor) in the forwards and backwards passes to be able to generate new data different from the historical ones.  

#### Results
Here are the results of the qq-plot :

<img width="474" alt="bnp" src="https://user-images.githubusercontent.com/119663180/205840023-6dcba43f-112b-4f91-98c7-f5e75625d41e.png">




## Contributors
Basile Hogenmuller alias @bashog
Simon Evanno alias @Simzer994
