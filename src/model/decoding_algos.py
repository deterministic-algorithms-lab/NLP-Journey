import jax
import jax.numpy as jnp
from functools import partial


def get_eval_fn(pure_logits_fn, task):

    @jax.jit
    def pure_eval_fn(params, key, token_ids):
        logits = pure_logits_fn.apply(params, key, 
                                      token_ids, training=False,
                                      task=task)
        return logits
    
    return pure_eval_fn        

@partial(jax.jit, static_argnums = (1,2,))
def greedy(key, pure_logits_eval_fn, max_length, params, initial_token_ids):
    
    token_ids = initial_token_ids
    for i in range(max_length):
        
        key, subkey = jax.random.split(key)
        logits = pure_logits_eval_fn(params, subkey, token_ids)

        predictions = jnp.argmax(logits[:,i,:], axis=-1)
    
        token_ids = jax.ops.index_update(token_ids, jax.ops.index[:,i+1],
                                         predictions)
    return token_ids