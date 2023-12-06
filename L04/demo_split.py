import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,100)
y = np.sin(x)+x**2

plt.scatter(x,y, s=1)
plt.xlim(0, 0.5)
plt.show()




