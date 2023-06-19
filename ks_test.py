import numpy as np
import scipy as sp
from scipy.stats import norm, kstest, skew

day, close = sp.loadtxt('No crisis (Days).csv',skiprows=1,delimiter=',',unpack= True)
returns = []

for i in range (0,(len(day)-1)):
    returns.append(close[i+1] - close[i])

returns.sort()
returns2 = (np.array(returns)).astype(int)

mean, sigma = norm.fit(returns2)
skewness = skew(returns2)

print('Mean is: ',mean)
print('Standard deviation is: ',sigma)
print('Skewness is: ',skewness)

ks_test= kstest(returns2,'norm',args=(10.624495712528178,114.24944512522094))
print(ks_test)