from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


class get_tokenizer():
    
    def __init__(self, config, data_files):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()
        self.trainer = BpeTrainer(special_tokens=['<s>', '</s>', '<unk>', '<pad>', '<mask>'])
        self.tokenizer.train(trainer, data_files)
        self.config = config
        self.set_up_tokenizer()
    
    def set_up_tokenizer(self):
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id('<pad>'),
                                      length=self.config['max_length'])
        
        self.tokenizer.enable_truncation(self.config['max_length']-1)

    def batch_encode_plus(self, batch, **kwargs):
        return self.tokenizer.encode_batch(batch, **kwargs)
