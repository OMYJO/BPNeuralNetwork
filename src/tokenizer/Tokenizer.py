class TokenizerV0(object):
    def __init__(self, max_len: int, vocab_file, cls_token=None, sep_token=None):
        self.max_len = max_len
        with open(vocab_file, 'r', encoding="utf-8") as f:
            self.vocab = list(f)
        self.cls_token = cls_token
        if self.cls_token is not None and self.cls_token not in self.vocab:
            raise ValueError("cls_token {} do not exist in vocabulary")
        self.sep_token = sep_token
        if self.sep_token is not None and self.sep_token not in self.vocab:
            raise ValueError("sep_token {} do not exist in vocabulary")
