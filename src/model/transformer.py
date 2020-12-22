import haiku as hk
import jax.numpy as jnp
import jax

class TransformerBlock(hk.Module):

    def __init__(self, config, is_autoregressive=False):
        super().__init__()
        self.config = config
        self.is_autoregressive = is_autoregressive
    
    def __call__(self, x, mask, training = False):

        attention_output = MultiHeadAttention(self.config, 
                                              self.is_autoregressive)(x, x, mask, training=training)
        
        residual = attention_output+x

        attention_output = hk.LayerNorm(axis=-1,
                                        create_scale=True,
                                        create_offset=True,)(residual)

        mlp_output = TransformerMLP(self.config)(attention_output, training=training)

        output_residual = mlp_output+attention_output

        layer_output = hk.LayerNorm(axis=-1,
                                    create_scale=True,
                                    create_offset=True,)(output_residual)
        
        return layer_output


class MultiHeadAttention(hk.Module):
    def __init__(self, config, is_autoregressive=False):
        super().__init__()
        self.config = config
        self.is_autoregressive = is_autoregressive
    
    def _split_into_heads(self, x):
        return jnp.reshape(x, [x.shape[0], x.shape[1], self.config['n_heads'], x.shape[2]//self.config['n_heads']])
    
    def get_attn_mask(self, seq_len):
        mask = jnp.ones([seq_len, seq_len])
        mask = jnp.triu(mask, k=1)
        return mask*-2**32
    
    def __call__(self, x, y, pad_mask, training=False):
        
        queries = hk.Linear(output_size=self.config['d_model'])(y)
        
        keys = hk.Linear(output_size=self.config['d_model'])(x)
        
        values = hk.Linear(output_size=self.config['d_model'])(x)
        
        queries = self._split_into_heads(queries)
        keys = self._split_into_heads(keys)
        values = self._split_into_heads(values)

        attention_logits = jnp.einsum('btnh,bsnh->bnts', queries, keys)
        attention_logits /= np.sqrt(queries.shape[-1])

        attention_logits += jnp.reshape(mask*-2**32, [mask.shape[0],1,1,mask.shape[1]])
        
        if self.is_autoregressive:
            attention_logits += self.get_attn_mask(y.shape[1])

        attention_weights = jax.nn.softmax(attention_logits, axis=-1)
        per_head_attention_output = jnp.einsum('bsnh,bnts->btnh', values, attention_weights)
        
        attention_output = jnp.reshape(per_head_attention_output, 
                                       [per_head_attention_output.shape[0], per_head_attention_output.shape[1], -1])

        attention_output = hk.Linear(output_size=self.config['d_model'])(attention_output)
        
        if training:
            attention_output = hk.dropout(rng=hk.next_rng_key(),
                                          rate=self.config['attention_drop_rate'],
                                          x=attention_output)
        
        return attention_output


def gelu(x):
    return x*0.5*(1.0+jax.scipy.special.erf(x / jnp.sqrt(2.0)))


class TransformerMLP(hk.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, x, training=False):

        intermediate_output = hk.Linear(output_size=self.config['intermediate_size'])(x)

        intermediate_output = gelu(intermediate_output)

        output = hk.Linear(output_size=self.config['d_model'])(intermediate_output)
        
        if training:
            output = hk.dropout(rng=hk.next_rng_key(),
                                rate=self.config['fully_connected_drop_rate'],
                                x=output)
        
        return output


class TransformerFeaturizer(hk.Module):
    
    def __init__(self, config, is_autoregressive=False):
        super().__init__()
        self.config = config
        self.is_autoregressive = is_autoregressive

    def __call__(self, token_ids, training=False):
        x = Embedding(self.config)(token_ids, training=training)
        
        mask = (jnp.bitwise_or(token_ids==self.config['pad_id'], 
                               token_ids==self.config['mask_id'])).astype(jnp.float32)
    
        for layer_num in range(self.config['n_layers']):
            x = TransformerBlock(config, 
                                 self.is_autoregressive)(x,mask,training)
        
        return x