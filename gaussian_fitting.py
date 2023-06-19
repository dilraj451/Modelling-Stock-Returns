import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import curve_fit
from scipy.stats import kstest,shapiro,anderson

day, close = sp.loadtxt('No crisis (Days).csv',skiprows=1,delimiter=',',unpack= True)
returns = []

for i in range (0,(len(day)-1)):
    returns.append((close[i+1]-close[i])/close[i])

returns.sort()
returns2 = (np.array(returns)).astype(float)
returns3 = np.array(returns).astype(float)
returns3.sort()

rounded_to = 0.003

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
plt.title('FTSE 250 Daily Returns (w/ COVID-19 crisis)')
plt.xlabel('Daily return (GBP)')
plt.ylabel('Frequency')
plt.hist(returns2,bins=np.arange(min(returns2), max(returns2) + binwidth, binwidth),align='right')
plt.errorbar(diff,freq,freq_error,fmt='none',ecolor='orange')

def f(x,a,b,c):
    return a * np.exp(-(x-b)**2/(2.0*c**2))
p_opt, p_cov = curve_fit(f,diff, freq, sigma=freq_error)
if (p_opt[2] < 0):
    p_opt[2] = -p_opt[2]
a,b,c = p_opt
best_fit_gauss_2 = f(diff,a,b,c)

x=np.linspace(min(diff),max(diff),100000)
y = p_opt[0]*np.exp(-((x-p_opt[1])**2)/(2*p_opt[2]**2))
plt.plot(x,y,color='red')
plt.show()

print('Amplitude: {} +\- {}'.format(p_opt[0], np.sqrt(p_cov[0,0])))
print('Mean: {} +\- {}'.format(p_opt[1], np.sqrt(p_cov[1,1])))
print('Standard Deviation: {} +\- {}'.format(p_opt[2], np.sqrt(p_cov[2,2])))

def reduced_chi_square(fit, x, y, yerr,N,n_param):
    return sum(((fit - y)/yerr)**2)/(N-n_param)

red_chi_squared = reduced_chi_square(best_fit_gauss_2, diff, freq, freq_error,len(freq),len(p_opt))
print('Reduced Chi-Squared: {}'.format(red_chi_squared))

ks_test= kstest(returns3,'norm',args=(p_opt[1],p_opt[2]))
print('Kolmogorov-Smirnoff test statistic: ',ks_test[0])
print('Kolmogorov-Smirnoff test p-value: ',ks_test[1])

shap_test_stat, shap_test_p  = shapiro(freq)
print('Shapiro-Wilk test statistic: ',shap_test_stat)
print('Shapiro-Wilk test p-value: ',shap_test_p)
anderson_darling = anderson(freq,dist='norm')

amend_and = anderson_darling[0]*(1 + (0.75/len(freq)) + (2.25/(len(freq)**2)))
ander_p = 0
if (amend_and >= 0.6):
    ander_p = np.exp(1.2937 - 5.709*amend_and + 0.0186*(amend_and**2))
elif (0.34 <= amend_and < 0.6):
    ander_p = np.exp(0.9177 - 4.279*amend_and - 1.38*(amend_and**2))
elif (0.2 < amend_and < 0.34):
    ander_p = 1-np.exp(-8.318 + 42.796*amend_and - 59.938*(amend_and**2))
else:
    ander_p = 1-np.exp(-13.436 + 101.14*amend_and - 223.73*(amend_and**2))
print('Anderson-Darling test statistic: ',amend_and)
print('Anderson-Darling test p-value: ',ander_p)
print(red_chi_squared*(len(freq)-len(p_opt)))
print(len(freq)-len(p_opt))