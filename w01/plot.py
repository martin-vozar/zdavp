import numpy as np
import matplotlib.pyplot as plt

def func1(xi):
    return(2*xi)

def func2(xi):
    xi = xi + 1
    return(2*xi)

x = np.linspace(0, 10, 11)

plt.scatter(x, func1(x))
plt.scatter(x, func2(x))

plt.show()