import tensorflow_datasets as tfds

def load_tf_dataset(config, training, split, n_epochs, n_examples, name='imdb_revies'):

    ds = tfds.load(name, 
                   split=f"{split}[:{n_examples}]").cache().repeat(n_epochs)
    
    if training:
        ds = ds.shuffle(10*config['batch_size'], seed=0)
    
    ds = ds.batch(config['batch_size'])

    return tfds.as_numpy(ds)

