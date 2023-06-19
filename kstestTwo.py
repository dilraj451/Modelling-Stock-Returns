import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import curve_fit
from scipy.stats import kstest, ks_2samp

dayNorm, closeNorm = sp.loadtxt('No crisis (Days).csv',skiprows=1,delimiter=',',unpack= True)
dayCrisis, closeCrisis = sp.loadtxt('Just 2008 (Days).csv',skiprows=1,delimiter=',',unpack= True)

returnsNorm = []
returnsCrisis = []

for i in range (0,(len(dayNorm)-1)):
    returnsNorm.append((closeNorm[i+1]-closeNorm[i])/closeNorm[i])
for k in range(0,len(dayCrisis)-1):
    returnsCrisis.append((closeCrisis[i+1]-closeCrisis[i])/closeCrisis[i])

returnsNorm.sort()
returnsCrisis.sort()

returns2 = np.array(returnsNorm).astype(float)
returns3 = np.array(returnsNorm).astype(float)
returns4 = np.array(returnsCrisis).astype(float)

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
print(ks_test)
print('Threshold for 95% confidence interval: ',(1.35810/np.sqrt(len(returns3))))

compare = ks_2samp(returns3,returns4)
print(compare)