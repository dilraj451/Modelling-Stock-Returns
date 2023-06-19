import numpy as np

vals1 = np.log10(np.array([3.64e-118,3.46e-230]))
vals2 = np.log10(np.array([1.29e-113,2.05e-219]))

diffs = vals1 - vals2

avg = np.mean(diffs)

st_dev = np.std(diffs)
error = st_dev/np.sqrt(len(diffs))
print('The average order of magnitude is: ', avg,' +/- ', error)