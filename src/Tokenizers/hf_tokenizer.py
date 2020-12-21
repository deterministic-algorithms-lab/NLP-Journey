from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing


class LM_Tokenizer:
    
    def __init__(self, config, data_files):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()
        self.trainer = BpeTrainer(special_tokens=['<s>', '</s>', '<unk>', '<pad>', '<mask>'])
        self.tokenizer.train(self.trainer, data_files)
        self.config = config
        self.set_up_tokenizer()
    
    def set_up_tokenizer(self):
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id('<pad>'),
                                      length=self.config['max_length'])
        
        self.tokenizer.enable_truncation(self.config['max_length']-1)

        self.tokenizer.post_processor = TemplateProcessing(single = "<s> $A </s>",
                                                           pair = "<s> $A </s> $B </s>",
                                                           special_tokens=[('<s>',1), ('</s>',2)])

    def decode_to_str(self, batch_text) :
        """
        Converts bytes string data to text. And truncates to max_len. 
        """
        max_len = self.config['max_length']
        return [ ' '.join(text.decode('utf-8').split()[:max_len]) if isinstance(text, bytes) else text[:max_len]
             for text in batch_text ]

    def batch_encode_plus(self, batch1, batch2=None):
        """
        Two batches correspond to sequences of different type/language.
        """
        if batch2 is None :
            return self.tokenizer.encode_batch( self.decode_to_str(batch1) )
        
        else :
            lis = [ (seq1,seq2) for seq1, seq2 in zip( self.decode_to_str(batch1), self.decode_to_str(batch2) ) ]
            return self.tokenizer.encode_batch(lis)
