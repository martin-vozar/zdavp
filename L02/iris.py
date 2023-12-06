import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def linear(xi, a, b):
    return(a*xi + b)

data = pd.read_csv('iris.csv')

setosa = data[data['variety'] == 'Setosa']

setosa_pw = setosa['petal.width']
setosa_pl = setosa['petal.length']

par, cov = curve_fit(linear, setosa_pw, setosa_pl)

plt.scatter(setosa_pw, setosa_pl)
plt.show()
