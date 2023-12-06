import time

import jax
from jax import random, numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
from flax.training import orbax_utils
import optax
import orbax.checkpoint
from train import train_epoch, eval_model

import pandas as pd
import matplotlib.pyplot as plt

start_time = time.time()

rng = jax.random.PRNGKey(1)
###########################################################################
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
x = np.array(x)

y = np.array(y)
y = jax.nn.one_hot(y, num_classes=3)

ds = {'input': np.array(x), 'label': np.array(y)}
ds['input'] = jax.random.permutation(rng, ds['input'])
ds['label'] = jax.random.permutation(rng, ds['label'])

train_ds = {'input' : ds['input'][:int(0.8*len(ds['input']))], 
            'label':  ds['label'][:int(0.8*len(ds['input']))]}
test_ds  = {'input' : ds['input'][int(0.8*len(ds['input'])):], 
            'label':  ds['label'][int(0.8*len(ds['input'])):]}

inTheShapeOf = random.normal(rng, np.shape(train_ds['input'][0]))
#######################################################################
print(f'Sample x data of shape {np.shape(x[0])}')
print(x[0])
print(x[int(len(x)/2)])
print(x[-1])

print(f'Sample y data of shape {np.shape(y[0])}')
print(y[0])
print(y[int(len(y)/2)])
print(y[-1])
#######################################################################
class demodel(nn.Module):
    @nn.compact
    def __call__(self, xi, train=False):
        for i in range(16):
            xi = nn.Dense(features=64)(xi)
            xi = nn.gelu(xi)
        #x = nn.Dropout(0.5, deterministic=not train)(x)
        xi = nn.Dense(features=3)(xi)
        xi = nn.softmax(xi)
        return xi
    
model = demodel()

############################################################################################

learning_rate = 3e-4
momentum = 0.8

rng, init_rng = random.split(rng)
init_params = model.init(init_rng, inTheShapeOf)['params']
tx = optax.adam(learning_rate, momentum)
state = train_state.TrainState.create(
        apply_fn=model.apply, params=init_params, tx=tx)
del init_rng
jax.tree_util.tree_map(lambda x: x.shape, init_params)

num_epochs = 400
batch_size = 32

print('Initializing training...')
for epoch in range(1, num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state = \
        train_epoch(state, train_ds, batch_size, epoch, input_rng, num_epochs)
    test_loss, test_accuracy = \
        eval_model(state, state.params, test_ds)
    if epoch%int(np.log(num_epochs)*2) == 0:
        print(f'test : epoch: \t {epoch} \t loss: \t {test_loss:.5f}, \t acc: {test_accuracy*100:.5f}')

print("Finished training.")
"""
print("Saving to checkpoint file")

ckpt = {'model': state}

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
orbax_checkpointer.save('./tmp/mi01', ckpt, save_args=save_args)
"""

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Timed at {elapsed_time}")
