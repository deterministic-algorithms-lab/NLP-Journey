import haiku as hk
import jax.numpy as jnp
import jax
import optax

def make_lr_schedule(warmup_percentage, total_steps):
    
    def lr_schedule(step):
        percent_complete = step/total_steps
        
        #0 or 1 based on whether we are before peak
        before_peak = jax.lax.convert_element_type((percent_complete<=warmup_percentage),
                                                   np.float32)
        #Factor for scaling learning rate
        scale = ( before_peak*(percent_complete/warmup_percentage)
                + (1-before_peak) ) * (1-percent_complete)
        
        return scale
    
    return lr_schedule


def get_adam_opt(config):
    total_steps = config['n_epochs']*(config['n_examples']//config['batch_size'])

    lr_schedule = make_lr_schedule(warmup_percentage=0.1, total_steps=total_steps)

    opt = optax.chain(
            optax.clip_by_global_norm(config['max_grad_norm']),
            optax.adam(learning_rate=config['learning_rate']),
            optax.scale_by_schedule(lr_schedule),
        )
    
    return opt