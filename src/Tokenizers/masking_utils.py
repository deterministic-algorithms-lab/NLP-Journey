import jax
import jax.numpy as jnp
import random

@jax.jit
def mask_batch_mlm(key, config, batch_token_ids):
    """
    Random replacement of tokens with mask or other tokens from vocabulary.
    batch_token_ids : numpy tensor of ints
    For MLM and TLM tasks. 
    """
    original_batch = batch_token_ids 
    
    replacable = (batch_token_ids!=config['pad_id']) * \
                 (batch_token_ids!=config['sos_id']) * \
                 (batch_token_ids!=config['eos_id'])

    key, subkey = jax.random.split(key)
    random_seq = jax.random.uniform(subkey, 
                                    [config['batch_size'], config['max_length']],
                                    maxval=1)
    
    batch_token_ids = jnp.where((random_seq<=0.15*0.8) * replacable,
                                config['mask_id'], 
                                batch_token_ids)
    
    key, subkey = jax.random.split(key)
    random_words = jnp.floor( jax.random.uniform(subkey, 
                                                 [config['batch_size'], config['max_length']], 
                                                 maxval=config['vocab_size']) )
    random_words = jnp.asarray(random_words, dtype=jnp.int16)
    
    batch_token_ids = jnp.where( (random_seq>0.15*0.8) * (random_seq<=0.15*0.9) * replacable, 
                                random_words, 
                                batch_token_ids)
    
    return batch_token_ids, original_batch

@jax.jit
def mask_batch_clm(key, config, batch_token_ids):
    """
    For CLM. Randomly masks each sequence in batch, after a certain length.
    First mask the logits, then use with logits[:,:-1].
    This will lead to similar loss for MLM and CLM. You can also use a scaling factor, instead.
    """
    lengths = jnp.sum(batch_token_ids!=config['pad_id'], axis=-1) 

    key, subkey = jax.random.split(key)
    random_seq = jax.random.uniform(subkey,
                                    shape=[config['batch_size'],],
                                    minval=1,
                                    maxval=lengths,)
    
    to_predict = jnp.asarray(jnp.floor(random_seq), dtype=jnp.int16)
    
    to_predict = jnp.expand_dims(to_predict, -1)

    max_len_array = jnp.arange(config['max_length'])
    
    masked_token_ids = jnp.where(max_len_array>=to_predict, 
                                config['pad_id'], batch_token_ids)
    
    mask_for_logits = (max_len_array<to_predict)

    return masked_token_ids, batch_token_ids[:,1:], mask_for_logits 