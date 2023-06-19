import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import curve_fit

day, close = sp.loadtxt('No crisis (Days).csv',skiprows=1,delimiter=',',unpack= True)
returns = []

for i in range (0,(len(day)-1)):
    returns.append(close[i+1] - close[i])
    i=i+1

returns.sort()
returns2 = (np.array(returns)).astype(int)

rounded_to = 2

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
plt.title('FTSE 250 Daily Returns (with COVID-19)')
plt.xlabel('Daily return (GBP)')
plt.ylabel('Frequency')
plt.hist(returns2,bins=np.arange(min(returns2), max(returns2) + binwidth, binwidth),align='left')
plt.errorbar(diff,freq,freq_error,fmt='none',ecolor='orange')

def f(x,a,b,c):
    return a * np.exp(-(x-b)**2/(2.0*c**2))
p_opt, p_cov = curve_fit(f,diff, freq, sigma=freq_error)
a,b,c = p_opt
best_fit_gauss_2 = f(diff,a,b,c)

x=np.linspace(-1000,1000,10000)
y = p_opt[0]*np.exp(-((x-p_opt[1])**2)/(2*p_opt[2]**2))
plt.plot(x,y,color='red')
plt.show()

print('Amplitude: {} +\- {}'.format(p_opt[0], np.sqrt(p_cov[0,0])))
print('Mean: {} +\- {}'.format(p_opt[1], np.sqrt(p_cov[1,1])))
print('Standard Deviation: {} +\- {}'.format(p_opt[2], np.sqrt(p_cov[2,2])))

def calc_reduced_chi_square(fit, x, y, yerr, N, n_param):
    return (1.0/(N-n_param))*sum(((fit - y)/yerr)**2)

reduced_chi_squared = calc_reduced_chi_square(best_fit_gauss_2, diff, freq, freq_error, len(freq), 3)
print('Reduced Chi Squared using scipy: {}'.format(reduced_chi_squared))
print('Number of degrees of freedom: {}'.format(len(freq)-3))