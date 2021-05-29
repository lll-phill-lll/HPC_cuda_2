import numpy as np
import matplotlib.pyplot as plt

img=[]
for line in open('res.txt', 'r'):
  img.append([float(s) for s in line.split()])


plt.imshow(img)
plt.savefig('laplas.png')
