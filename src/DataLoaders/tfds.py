import tensorflow_datasets as tfds

def load_tf_dataset(config, training, split, n_epochs, n_examples, name='imdb_reviews', data_dir='./'):

    ds = tfds.load(name, 
                   split=f"{split}[:{n_examples}]",
                   data_dir=data_dir).cache().repeat(n_epochs)
    
    if training:
        ds = ds.shuffle(10*config['batch_size'], seed=0)
    
    ds = ds.batch(config['batch_size'])

    return tfds.as_numpy(ds)

