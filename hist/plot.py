import numpy as np
import matplotlib.pyplot as plt

res = np.loadtxt('res.txt')

plt.plot(np.linspace(0, 256, 256), res)
plt.show()
plt.savefig('hist.png')
