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
            words = x["hero"]
            positions = x["pos"]
            types = x["type"]
            # 添加特殊token
            if self.sep_token is not None:
                words = [self.vocab.index(self.sep_token)] * 2 + words
                positions = [-100] * 2 + positions
                types = [0, 1] + types
            if self.cls_token is not None:
                words = [self.vocab.index(self.cls_token)] + words
                positions = [-100] + positions
                types = [4] + types
            # 全局bp的信息补全
            if is_global_ban_pick and given_global_ban_pick:
                for j in range(i):
                    for k in range(len(x["hero"])):
                        if match[j]["type"][k] == 0:
                            that_team = match[j]["blue"]
                        elif match[j]["type"][k] == 1:
                            that_team = match[j]["red"]
                        else:
                            continue
                        if that_team == x["blue"]:
                            words.append(self.vocab.index(match[j]["hero"][k]))
                            positions.append(-100)
                            types.append(5)
                        elif that_team == x["red"]:
                            words.append(self.vocab.index(match[j]["hero"][k]))
                            positions.append(-100)
                            types.append(6)
                        else:
                            raise ValueError("The team {} is a error team!".format(that_team))
            # 位置编码整理
            pos_set = list(set(positions))
            pos_set.sort()
            pos_dict = dict(zip(pos_set, range(len(pos_set))))
            positions = [pos_dict[p] for p in positions]
            # 序列过长截断
            assert len(words) == len(positions) == len(types)
            if len(words) > self.max_len:
                words = words[:self.max_len]
                positions = positions[:self.max_len]
                types = positions[:self.max_len]
            # 装车
            r.append((words, positions, types))
        return r

if __name__ == '__main__':
    import json
    import os
    with open(os.path.join("data","1.json"),"r",encoding="utf-8") as f
        data = json.load(f)

