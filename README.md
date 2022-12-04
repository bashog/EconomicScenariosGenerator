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

The Time series class defines 7 functions that we will to process and generate the results :
- __init__
- pre_processing
- plot
- statistics
- correlation
- bootstrap_esg
- rbm_esg

### Economic scenario generator abstract class
- purpose of the class
- design of the class

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

