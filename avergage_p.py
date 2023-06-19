import numpy as np

vals = np.log10(np.array([1.29e-113,2.05e-219]))

avg = np.mean(vals)

st_dev = np.std(vals)
error = st_dev/np.sqrt(len(vals))
print('The average order of magnitude is: ', avg,' +/- ', error)