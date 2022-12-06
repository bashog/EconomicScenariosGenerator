# Economic Scenario Generator

## Context

## Project motivations and purposes

During the project, we will work on exchange rate returns from past years and then use different methods for generating economic scenarios :
- Bootstrapping
- Restricted Boltzmann Machine (RBM)

## Structure of the code
- bootstrap contains the bootstrapping class. It allows to call Python functions which are going to generate bootstrap samples.
- esg contains the ESG class. It is the blueprint for the inheritance of the different ESG classes. It defines Python functions used by all the different methods.
- rbm contains the RBM and RBM_simple classes. It allows to call Python functions which are going to generate economic scenarios with RBM methods
- statistics_tools defines Python functions which give information about statistical properties and tests.
- timeseries contains the Timeseries class. This class uses the other classes and functions to get the results. It is the main running file.
- utils defines Python functions about returns, generation of samples or plot.

## The models

### Time series class

The Time series class defines 7 functions that we will use on the source data to process and generate the results :
    - __init__ : initialize the class and keep the data needed;
    - pre_processing : basic preprocessing of the data by choosing the way the missing values are filled, the method used to calculate the returns and whether the extreme values are kept ;
    - plot : plot the prices or the returns ;
    - statistics : compute the main statistics of the returns ;
    - correlation : calculate the correlation matrix ;
    - bootstrap_esg : generate the bootstrap samples ;
    - rbm_esg : generate the rbm samples.
It is the class called to generate results in our notebook.


### Economic scenario generator abstract class

The abstract class for the Economic Scenario Generator is the blueprint for the inheritance of the different ESG classes. In our case, we used it for the bootstrap and the rbm classes.
The class must implement the following functions:
    - pre_processing : Preprocess the data ;
    - train : Train the model on the train set ;
    - generate : Generate the scenarios ;
    - quantiles : Get the quantiles of the generated data ;
    - correlation : Get the correlation matrix of the generated data ;
    - plot_returns : Plot the returns of the initial data and the quantiles of the generated data.


### Bootstrapping
- explanation of the algorithm
- how to use it (snippet of code)
- results (photo etc)


### Restricted Boltzmann Machine
- explanation of the algorithm
- how to use it
- results



## Contributors
Basile Hogenmuller alias @bashog
Simon Evanno alias @Simzer994
Viviane Feng alias @vivianefeng

