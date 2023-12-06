import jax
import jax.numpy as jnp
import numpy as np
import optax

@jax.jit
def cross_entropy_loss(*, logits, labels):
    return optax.softmax_cross_entropy(logits=logits, labels=labels).mean()

@jax.jit
def mse(logits, labels):
    return jnp.mean((logits - labels) ** 2) / 2.0

floss = mse

def compute_metrics(*, logits, labels):
    # assume labels to be onehot(true_labels)
    loss = floss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics

@jax.jit
def train_step(state, batch):
    """Train for a single batch."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['input'])
        loss = floss(logits=logits, labels=batch['label'])
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch['label'])
    return state, metrics

@jax.jit
def eval_step(state, params, batch):
    logits = state.apply_fn({'params': params}, batch['input'])
    return compute_metrics(logits=logits, labels=batch['label'])

def eval_model(state, params, test_ds):
    metrics = eval_step(state, params, test_ds)
    metrics = jax.device_get(metrics)
    summary = jax.tree_map(lambda x: x.item(), metrics)
    return summary['loss'], summary['accuracy']

def train_epoch(state, train_ds, batch_size, epoch, rng, num_epochs=400):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['input'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics = train_step(state, batch)
        batch_metrics.append(metrics)
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]}
    if epoch%int(np.log(num_epochs)*2) == 0:
        print(f"train: epoch: \t {epoch} \t loss: \t {epoch_metrics_np['loss']:.5f}, \t acc: {epoch_metrics_np['accuracy']*100:.5f}")  
    return state