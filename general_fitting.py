import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from collections import Counter
from lmfit.models import GaussianModel

day, close = sp.loadtxt('No crisis (Days).csv',skiprows=1,delimiter=',',unpack= True)
returns = []

for i in range (0,(len(day)-1)):
    returns.append(close[i+1] - close[i])
    i=i+1

returns.sort()
returns2 = (np.array(returns)).astype(int)

rounded_to = 40

for j in range (0,(len(returns2)-1)):
    if (returns2[j]%rounded_to) >= (rounded_to/2):
        returns2[j] = returns2[j] + (rounded_to-(returns2[j]%rounded_to))
    else:
        returns2[j] = returns2[j] - (returns2[j]%rounded_to)

count = Counter(returns2)
dtype = dict(names = ['id','data'], formats=['i8','i8'])
array = np.fromiter(iter(count.items()), dtype=dtype)
counted = np.array(list(count.items()))
difference, frequency = np.hsplit(counted,2)

diff = difference.flatten()
freq = frequency.flatten()

freq_error = np.sqrt(freq)

binwidth = rounded_to
plt.title('FTSE 250 Daily Returns (inc COVID-19 crisis)')
plt.xlabel('Daily return (GBP)')
plt.ylabel('Frequency')
plt.hist(returns2,bins=np.arange(min(returns2), max(returns2) + binwidth, binwidth),align='left')
plt.errorbar(diff,freq,freq_error,fmt='none',ecolor='orange')