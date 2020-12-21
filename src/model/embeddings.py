import haiku as hk
import jax.numpy as jnp
import jax


class Embedding(hk.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config


    def __call__(self, token_ids, lang_ids=None, training=False):
        """
        token_ids: ints of shape (batch, n_seq)
        """
        
        flat_token_ids = jnp.reshape(token_ids, [-1])
        
        flat_token_embeddings = hk.Embed(vocab_size=self.config['vocab_size'],
                                         embed_dim=self.config['d_model'])(flat_token_ids)

        token_embeddings = jnp.reshape(flat_token_embeddings, [token_ids.shape[0], -1, self.config['d_model']])
        
        embeddings = token_embeddings + PositionEmbeddings(self.config)()
        
        if lang_ids is not None:
            embeddings += LanguageEmbeddings(self.config)(lang_ids)
        
        embeddings = hk.LayerNorm(axis=-1,
                                  create_scale=True,
                                  create_offset=True,)(embeddings)
        if training:
            embeddings = hk.dropout(hk.next_rng_key(),
                                    rate=self.config['embed_dropout_rate'],
                                    x=embeddings)
        
        return embeddings


class PositionEmbeddings(hk.Module):
    """
    A position embedding of size [max_seq_leq, word_embedding_dim]
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.offset = 0

    def get_init_pe(self):
        
        pe = np.zeros([self.config['max_length'], self.config['d_model']])
        
        position = np.arange(0, self.config['max_length']).reshape(-1,1)
        
        div_term = np.exp(np.arange(0, self.config['d_model'],2)*
                          -np.log(10000.0)/self.config['d_model'])
        
        pe[:, 0::2] = np.sin(position*div_term)
        pe[:, 1::2] = np.cos(position*div_term)
        
        return pe


    def __call__(self):
        
        position_weights = hk.get_parameter("position_embeddings",
                                            [self.config['max_length'], self.config['d_model']],
                                            init=hk.initializers.Constant(self.get_init_pe()))
        
        start = self.offset
        end = self.offset+self.config['max_length']
        
        return position_weights[start:end]


class LanguageEmbeddings(hk.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, lang_ids):

        return hk.Embed(vocab_size=len(self.config['lang2id'])+1, 
                        embed_dim=self.confid['d_model'])(lang_ids)