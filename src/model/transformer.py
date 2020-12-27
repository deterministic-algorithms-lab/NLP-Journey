import haiku as hk
import jax.numpy as jnp
import jax
from src.model.embeddings import Embedding

class TransformerBlock(hk.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def __call__(self, x, mask, training=False, is_autoregressive=False):

        attention_output = MultiHeadAttention(self.config)(x, x, mask,
                                                           training=training, is_autoregressive=is_autoregressive)
        
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

class TransformerDecoderBlock(hk.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def __call__(self, y, tgt_mask, src_mask, x_embds, training=False):

        attention_output = MultiHeadAttention(self.config)(y, y, tgt_mask,
                                                           training=training, is_autoregressive=True)
        
        residual = attention_output+y

        self_attention_output = hk.LayerNorm(axis=-1,
                                             create_scale=True,
                                             create_offset=True,)(residual)
        
        attention_output = MultiHeadAttention(self.config)(x_embds, self_attention_output, src_mask,
                                                           training=training, is_autoregressive=False)
        
        residual = attention_output+self_attention_output

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
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def _split_into_heads(self, x):
        return jnp.reshape(x, [x.shape[0], x.shape[1], self.config['n_heads'], x.shape[2]//self.config['n_heads']])
    
    def get_attn_mask(self, seq_len):
        mask = jnp.ones([seq_len, seq_len])
        mask = jnp.triu(mask, k=1)
        return mask*-2**32
    
    def __call__(self, x, y, mask, training=False, is_autoregressive=False):
        
        queries = hk.Linear(output_size=self.config['d_model'])(y)
        
        keys = hk.Linear(output_size=self.config['d_model'])(x)
        
        values = hk.Linear(output_size=self.config['d_model'])(x)
        
        queries = self._split_into_heads(queries)
        keys = self._split_into_heads(keys)
        values = self._split_into_heads(values)

        attention_logits = jnp.einsum('btnh,bsnh->bnts', queries, keys)
        attention_logits /= jnp.sqrt(queries.shape[-1])

        attention_logits += jnp.reshape(mask*-2**32, [mask.shape[0],1,1,mask.shape[1]])
        
        if is_autoregressive:
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
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def get_mask(self, token_ids):
        return (jnp.bitwise_or(src_token_ids==self.config['pad_id'], 
                                   src_token_ids==self.config['mask_id'])).astype(jnp.float32)
    
    def __call__(self, token_ids, lang_ids=None, training=False, is_autoregressive=False):
        
        x = Embedding(self.config)(token_ids, lang_ids=lang_ids, training=training)
        
        mask = self.get_mask(token_ids)

        for layer_num in range(self.config['n_layers']):
            x = TransformerBlock(self.config)(x, mask,
                                         training=training, is_autoregressive=is_autoregressive)
        
        return x

class LogitsTransformer(hk.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, token_ids, lang_ids=None, training=False, is_autoregressive=False):
        x = TransformerFeaturizer(self.config)(token_ids, lang_ids, 
                                               training=training, is_autoregressive=is_autoregressive)
        logits = hk.Linear(output_size=self.config['vocab_size'])(x)
        return logits

class VaswaniTransformer(hk.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def get_mask(self, token_ids):
        return (jnp.bitwise_or(src_token_ids==self.config['pad_id'], 
                                   src_token_ids==self.config['mask_id'])).astype(jnp.float32)
        
    def __call__(self, src_token_ids, tgt_token_ids, src_lang_ids=None, tgt_lang_ids=None, training=False):
        
        x_embds = TransformerFeaturizer(self.config)(src_token_ids, lang_ids=src_lang_ids,
                                                    training=True)
        
        src_mask = self.get_mask(src_token_ids)
        tgt_mask = self.get_mask(tgt_token_ids)

        y = Embedding(self.config)(tgt_token_ids, lang_ids=tgt_lang_ids, training=training)

        tgt_features = TransformerDecoderBlock(self.config)(y, tgt_mask, src_mask, x_embds, training=training)

        logits = hk.Linear(output_size=self.config['tgt_vocab_size'])(tgt_features)

        return logits