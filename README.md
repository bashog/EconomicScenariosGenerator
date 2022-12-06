# Economic Scenario Generator

## Context

## Project motivations and purposes


## Structure of the code
This repository contains an abstract mother class for generating economic scenarios, as well as two child classes that inherit from the mother class: a `Bootstrap` class and a Restricted Boltzmann Machine `RBM` class.
These two classes are called by the `TimeSeries' class, which is used to train the generator and generate new scenarios.

### Usage
To use the `EconomicScenarioGenerator` class, you will need to create a child class that implements the abstract methods defined in the mother class. You can then use the child class to train the generator to historical returns, generate new scenarios, and evaluate the performance of the generator by comparing the historical quantiles and the quantiles generated.

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
- explanation of the algorithm
- how to use it
- results



## Contributors
- Basile Hogenmuller alias @bashog
- Simon Evanno alias @Simzer994
