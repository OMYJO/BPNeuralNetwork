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

    def tokenize(self, match, is_global_ban_pick=False, given_global_ban_pick=True):
        r = []
        for i, x in enumerate(match):
            words = []
            positions = []
            types = []
            cnt = 0
            if self.cls_token is not None:
                words.append(self.vocab.index(self.cls_token))
                positions.append(0)
                types.append(0)
            if self.sep_token is not None:
                words.append(self.vocab.index(self.sep_token))
                positions.append(0)
                types.append(1)
                words.append(self.vocab.index(self.sep_token))
                positions.append(0)
                types.append(2)
            cnt += 1
            if i < 6 or not is_global_ban_pick:
                template_camp___ = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
                template_banpick = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
                assert len(template_camp___) == len(template_banpick)
                cnt_banpick = [0, 0, 0, 0]
                for j in range(len(template_camp___)):
                    if template_camp___[j] == 0:
                        camp = x.blue
                    else:
                        camp = x.red
                    if template_banpick[j] == 0:
                        banpick = camp.pick
                    else:
                        banpick = camp.ban
                    idx = 2 * template_camp___[j] + template_banpick[j]
                    if len(banpick) > cnt_banpick[idx]:
                        words.append(self.vocab.index(banpick[cnt_banpick[idx]]))
                        cnt_banpick[idx] += 1
                        positions.append(cnt)
                        cnt += 1
                        types.append(template_camp___[j] + 1 + 2 * template_banpick[j])

                if is_global_ban_pick and given_global_ban_pick:
                    for j in range(i):
                        pass
            else:
                for word in x.blue.pick:
                    words.append(self.vocab.index(word))
                    positions.append(0)
                    types.append(5)
                for word in x.red.pick:
                    words.append(self.vocab.index(word))
                    positions.append(0)
                    types.append(6)

            r.append((words, positions, types))
        return r
