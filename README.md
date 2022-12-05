# Economic Scenario Generator

## Context

## Project motivations and purposes


## Structure of the code


## The models

### Time series class

### Economic scenario generator abstract class
- purpose of the class
- design of the class

### Bootstrapping

#### Bootstrap method

Statistics science is based on learning from data. Statistical methods are therefore essential to make decisions and predictions while the situation already occured. The traditional approach (or large sample approach) consists of drawing one sample of size n from the data and that sample is used to calculate the data estimated to make inferences on. But this method tends to take into account outliers. In order to tackle this issue, the bootstrapping method is a statistical procedure that resamples a single data to create many simulated samples. In our case, we are considering the past datas of each currency and we draw randomly as many as returns we need for each scenarios (for example : 100).

The block samples are chosen randomly to create bootstrap resamples, there are several ways to choose these blocks :

- Simple block bootstrap: we choose from blocks of a fixed length, delta_block.
- Moving block bootstrap: slightly more complicated since it allows to overlap of the blocks.

#### Analysis

Here, we analyse daily, weekly and monthly frequencies. To make these analysis, we have chosen to plot from 01-01-2020 by chosing 100 scenarios and 04-09-2021 as test date.
For each type of frequency we plot the correlation between the USD, the JPY, the GBP and the CAD currency.

#### Example 

This is a boostrap code example that corresponds to the daily frequency :

```shell
test_dates = '2021-09-04'
plot_from='2020-01-01'
scenarios = 100
ts.bootstrap_esg(scenarios=scenarios, test_date=test_dates, plot_from=plot_from)
```
Which gives us for the USD currency : 
![](img/usd.PNG)


### Restricted Boltzman Machine
- explanation of the algorithm
- how to use it
- results



## Contributors
Basile Hogenmuller alias @bashog
Simon Evanno alias @Simzer994
Basma Bazi alias @basmabazi
