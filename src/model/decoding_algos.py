import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums = (1,2,3))
def greedy(key, pure_logits_fn, max_length, task, params, initial_token_ids):
    
    token_ids = initial_token_ids
    for i in range(max_length):
        
        key, subkey = jax.random.split(key)
        logits = pure_logits_fn.apply(params, subkey, token_ids, training=False, task=task)

        predictions = jnp.argmax(logits[:,i,:], axis=-1)
    
        token_ids = jax.ops.index_update(token_ids, jax.ops.index[:,i],
                                     predictions)
    return token_ids