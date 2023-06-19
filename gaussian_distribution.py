import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from collections import Counter

day_20, close_20 = sp.loadtxt('Combined 2008-2020 (Days).csv',skiprows=1,delimiter=',',unpack= True)
diff_20 = []

for i in range (0,(len(day_20)-1)):
    diff_20.append((close_20[i+1] - close_20[i])*100/close_20[i])
    i=i+1
diff_20.sort()

diff_20_rounded = [round(num*2)/2 for num in diff_20]

count = Counter(diff_20_rounded)
dtype = dict(names = ['id','data'], formats=['i8','i8'])
array = np.fromiter(iter(count.items()), dtype=dtype)
final_20 = np.array(list(count.items()))
percent, freq = np.hsplit(final_20,2)
percent = percent.astype(float)
print(percent)

mean = sum(percent*freq)/sum(freq)
sigma = np.sqrt(sum((percent - mean)**2)/sum(freq))
amp = np.amax(freq)

x=np.linspace(-10,10,1000)
y = amp*np.exp(-(x-mean)**2/(2*sigma**2))
plt.plot(x,y)
plt.show()

print(mean)
print(sigma)
print(amp)