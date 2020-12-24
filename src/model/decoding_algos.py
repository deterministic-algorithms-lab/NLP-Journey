import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnums = (1,2,3,))
def greedy(key, pure_logits_fn, config, params, initial_token_ids, *args, **kwargs):
    
    token_ids = initial_token_ids
    for i in range(config['max_length']):
        
        key, subkey = jax.random.split(key)
        logits = pure_logits_fn.apply(params, subkey, token_ids, *args, **kwargs)
    
        predictions = jnp.argmax(logits[:,i,:], axis=-1)
    
        token_ids = jax.ops.index_update(token_ids, jax.ops.index[:,i,:],
                                     predictions)
    return token_ids


    