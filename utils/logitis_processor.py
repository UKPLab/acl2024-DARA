import transformers
from torch import LongTensor, FloatTensor
import numpy as np

class _logitsProcessor(transformers.LogitsProcessor):
    def __init__(self, stop_words, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.eos_token_id = tokenizer.eos_token_id
        self.previous_ids = None
    
    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
    #    input_ids[:,:s]
        input_ids = input_ids[:,-5:]
        strings = self.tokenizer.batch_decode(input_ids)
        finished_status = np.array(list(map(lambda x: self.find_stop_words(x), strings)))

        # find finsihed sequences
        # set the scores of eos in infinished sequences to inf
        scores[finished_status != -1, self.eos_token_id] = 10000
        return scores

    def find_stop_words(self, string):
        # ipdb.set_trace()
        for stp in self.stop_words:
            pos = string.rfind(stp)
            if pos != -1:
                return pos
        # string can find any stop words in the list
        return -1 