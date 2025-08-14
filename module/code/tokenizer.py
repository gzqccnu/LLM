import re

class Tokenizer:
    # Note here: the corpus not the raw text, it should be preprocessed.
    def __init__(self, corpus):
        self.token_id_dict = {token: id for id, token in enumerate(sorted(set(corpus)))}
        self.id_token_dict= {id: token for id, token in enumerate(sorted(set(corpus)))}
    
    def encode(self, text):
        # preprogress the text
        preprocess = re.split('([.,\'\"?!:;()]|--|\s)', text)
        
        # remove the token with blank like this: __hello____world__.
        # here use '_' to replace blank
        preprocess = [
            item.strip() for item in preprocess if item.strip()
        ]

        # Add "<|unk|>" token
        preprocess = [
            item if item in self.token_id_dict
            else "<|unk|>" for item in preprocess
        ]

        # generate ids
        ids = [self.token_id_dict[token] for token in preprocess]

        return ids
    
    def decode(self, ids):
        # this function will join the blank string` to the list iterablly.
        # that will automatically to let token final sentence to have blank
        # nor it generation will like `whatcanIholdyouwith`
        text = " ".join([self.id_token_dict[id] for id in ids])

        # remove blank in front of punctuation mark
        text = re.sub(r'\s+([.,?!;:\'\"])', r'\1', text)

        return text