### Modelling Stock Returns

## Purpose
To determine whether the distribution of FTSE 250 stock returns could be modelled using a defined statistical distribution (e.g. Gaussian, Lorentzian, Laplacian etc) during periods of stability and crisis periods (2008 financial crisis and 2020 COVID-19 pandemic) respectively.

## Methods
Data on FTSE 250 index prices, during the respective time periods, were stored in csv files and cleaned on MS Excel. The data was imported into numerous python scripts where scipy and matplotlib.pyplot modules were used to vizualize (in histograms), fit, and assess the validity of the applied statistical model to the stock return data.

## User Instructions
Run any the following files to view the modelled distrbutions and results of statistical tests applied to assess the accuracy of each model: gaussian_fitting.py, laplacian.py, log_normal.py, lorentzian.py, moffat_fitting.py.

Alternatively, the histograms and applied models can be viewed in the existing png images in the folder.

## Conclusions
Using a 95% confidence interval, none of the statistical distrubutions sufficiently modelled the stock returns data; primarily due to the relatively high frequency of outliers particularly during crises. More complex models (e.g. modulus of sinc function) may be sufficent and should be examined in future investigations.