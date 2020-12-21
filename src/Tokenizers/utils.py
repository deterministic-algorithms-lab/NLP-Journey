def decode_to_str(batch_text, max_len=512) :
    """
    Converts bytes string data to text. And truncates to max_len. 
    """
    return [ ' '.join(text.decode('utf-8').split()[:max_len]) if isinstance(text, bytes) else text[:max_len]
             for text in batch_text ]

def concat_and_decode(train_batch1, train_batch2, max_len=512, sep_tok='</s>'):
    """
    For concatenating and decoding two batches of parallel sentences for TLM.
    """
    max_len = (max_len//2) - 1

    batch_text = [
                  (' '.join(text1.decode('utf-8').split()[:max_len]) if isinstance(text1, bytes) else text1[:max_len]) + sep_token +
                  (' '.join(text2.decode('utf-8').split()[:max_len]) if isinstance(text2, bytes) else text2[:max_len])
                  for text1, text2 in zip(train_batch1, train_batch2)
    ]

    return batch_text
