import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import curve_fit
from scipy.stats import kstest

day, close = sp.loadtxt('Just COVID (Days).csv',skiprows=1,delimiter=',',unpack= True)
returns = []

for i in range (0,(len(day)-1)):
    returns.append((close[i+1]-close[i])/close[i])

returns.sort()
returns2 = (np.array(returns)).astype(float)
returns3 = np.array(returns).astype(float)
returns3.sort()

rounded_to = 0.005

for j in range (0,len(returns2)):
    if (returns2[j]%rounded_to) >= (rounded_to/2):
        returns2[j] = returns2[j] + (rounded_to-(returns2[j]%rounded_to)) 
    else:
        returns2[j] = returns2[j] - (returns2[j]%rounded_to)
    if (-1e-10 <= returns2[j] <= 1e-10):
        returns2[j] = 0

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
plt.hist(returns2,bins=np.arange(min(returns2), max(returns2) + binwidth, binwidth),align='right')
plt.errorbar(diff,freq,freq_error,fmt='none',ecolor='orange')

def f (x,a,b,c):
    return a*np.exp(-abs(x-b)/c)

p_opt, p_cov = curve_fit(f,diff, freq, sigma=freq_error) 
a,b,c = p_opt
best_fit_gauss_2 = f(diff,a,b,c)

x=np.linspace(min(diff),max(diff),100000)
y = p_opt[0]*np.exp(-abs(x-p_opt[1])/p_opt[2])
plt.plot(x,y,color='red')
plt.show()

print('Amplitude: {} +\- {}'.format(p_opt[0], np.sqrt(p_cov[0,0])))
print('Mean: {} +\- {}'.format(p_opt[1], np.sqrt(p_cov[1,1]))) 
print('Standard Deviation: {} +\- {}'.format(p_opt[2], np.sqrt(p_cov[2,2])))

def reduced_chi_square(fit, x, y, yerr,N,n_param):
    return sum(((fit - y)/yerr)**2)/(N-n_param)

red_chi_squared = reduced_chi_square(best_fit_gauss_2, diff, freq, freq_error,len(freq),len(p_opt))
print('Reduced Chi-Squared: {}'.format(red_chi_squared))

ks_test= kstest(returns3,'laplace',args=(p_opt[1],p_opt[2]))
print('Kolmogorov-Smirnoff test statistic: ',ks_test[0])
print('Kolmogorov-Smirnoff test p-value: ',ks_test[1])