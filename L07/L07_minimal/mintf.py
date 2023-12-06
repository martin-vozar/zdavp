import time

import tensorflow as tf
import numpy as np
import pandas as pd

start_time = time.time()

df = pd.read_csv("../iris.csv")
df['petal.dist'] = (df['petal.length']**2+df['petal.width']**2)**.5

features = ['sepal.length',
            'sepal.width',
            'petal.dist',]
labels   = 'variety'

x = df[features]
y = df[labels]
d = {'Setosa' : 0,
     'Versicolor' : 1,
     'Virginica' : 2}
y = y.map(d)

nbatch = 64
nepochs = 400

model = tf.keras.Sequential([
     tf.keras.Input(shape=(3)),
     tf.keras.layers.Dense(64, activation='gelu'),
     tf.keras.layers.Dense(64, activation='gelu'),
     tf.keras.layers.Dense(64, activation='gelu'),
     tf.keras.layers.Dense(64, activation='gelu'),
     tf.keras.layers.Dense(64, activation='gelu'),
     tf.keras.layers.Dense(64, activation='gelu'),
     tf.keras.layers.Dense(64, activation='gelu'),
     tf.keras.layers.Dense(64, activation='gelu'),
     tf.keras.layers.Dense(64, activation='gelu'),
     tf.keras.layers.Dense(64, activation='gelu'),
     tf.keras.layers.Dense(64, activation='gelu'),
     tf.keras.layers.Dense(64, activation='gelu'),
     tf.keras.layers.Dense(64, activation='gelu'),
     tf.keras.layers.Dense(64, activation='gelu'),
     tf.keras.layers.Dense(64, activation='gelu'),
     tf.keras.layers.Dense(64, activation='gelu'),
     tf.keras.layers.Dropout(0.5), # will come back to this
     # i think it's what he said :D ^
     tf.keras.layers.Dense(3, activation="softmax"),])

opt = tf.keras.optimizers.Adam(learning_rate=3e-4, )

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.build((nbatch,len(features)))

model.fit(x,
          tf.keras.utils.to_categorical(y, num_classes=3), 
          batch_size=nbatch, epochs=nepochs,
          verbose=1)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Timed at {elapsed_time}")