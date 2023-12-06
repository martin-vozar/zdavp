import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("../iris.csv")
d = {'Setosa' : 0,
     'Virginica' : 1,
     'Versicolor' : 2}

p = df['sepal.length']
q = df['sepal.width']
r = df['petal.length']
s = df['petal.width']
t = df['variety']
t = t.map(d)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(p, 
                     q, 
                     r, 
                     s=s*10, 
                     c=t, 
                     cmap='viridis', alpha=0.9)

# Set axis labels
ax.set_xlabel("Curb Weight")
ax.set_ylabel("Horsepower")
ax.set_zlabel("Price")

# Add a colorbar
colorbar = plt.colorbar(scatter)
colorbar.set_label('City MPG')

# Show the plot
plt.show()