import random
import torch
import os
import json
from torch.utils.data import dataloader
from module.Tokenizer import TokenizerV0
from dataset.DataSet import ListDataSetV0


def data_loader():
    vocab = os.path.join("../", "..", "models", "version0", "vocab.txt")
    tokenizer = TokenizerV0(64, vocab, "[CLS]", "[SEP]")
    vocab_len = len(tokenizer.vocab)
    data_set = []
    basic_path = os.path.join("..", "..", "data")
    for file_name in os.listdir(basic_path):
        if "json" in file_name:
            with open(os.path.join(basic_path, file_name), "r", encoding="utf-8") as f:
                matchs = list(json.load(f))
                for match in matchs:
                    data_set += tokenizer.tokenize(match, match[0]["is_overallBP"], True)
    train_set = ListDataSetV0(data_set)
    train_loader = dataloader.DataLoader(train_set, batch_size=6, shuffle=True, collate_fn=lambda x: x)
    return train_loader


def make_attention_mask(seq_self, device):
    max_len_q = max(len(seq) for seq in seq_self)
    attention_mask = torch.zeros([len(seq_self), max_len_q, max_len_q], device=device)  # divice
    for i, seq in enumerate(seq_self):
        # for j in range(len(seq)):
        #     for k in range(len(seq)):
                attention_mask[i, :len(seq), :len(seq)] = 1
    return attention_mask


def data_generate(data_batch, vocab_len=120, device="cpu"):
    mask_rule = lambda rand: 3 if rand < 0.8 else (
        competition[0][hero_id] if rand > 0.9 else random.randint(16, vocab_len))
    mask = []
    train_dic = {}
    hero_battle = []
    position_battle = []
    type_token_battle = []
    old_hero = []
    max_len_q = max(len(seq[0]) for seq in data_batch)
    for i, competition in enumerate(data_batch):  # batch_size * 3 * m
        hero = []
        old_hero.append(competition[0])
        for hero_id in range(len(competition[0])):
            # for hero_id in range(max_len_q):
            if any(competition[2][hero_id] == xx for xx in (0, 1, 2, 3)) and (competition[0][hero_id] >= 16):
                if random.random() < 0.15:
                    mask.append((i, hero_id, competition[0][hero_id]))
                    hero.append(mask_rule(random.random()))
                else:
                    hero.append(competition[0][hero_id])
            else:
                hero.append(competition[0][hero_id])
        hero_battle.append(hero + [0]*(max_len_q - len(competition[0])))
        position_battle.append(competition[1] + [0]*(max_len_q - len(competition[0])))
        type_token_battle.append(competition[2] + [0]*(max_len_q - len(competition[0])))
        # battle = (hero, competition[1], competition[2])
        # batch.append(battle)
        # ground_truth.append(competition[0])
        # mask_position.append(mask)

    train_dic["input_ids"] = torch.tensor(hero_battle, device=device).long()
    train_dic["attention_mask"] = make_attention_mask(old_hero, device)
    train_dic["position_ids"] = torch.tensor(position_battle, device=device).long()
    train_dic["token_type_ids"] = torch.tensor(type_token_battle, device=device).long()
    return train_dic, mask


if __name__ == "__main__":
    train_loader = data_loader()
    # train_loader = [[[[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]],[[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]],[[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]]]]
    for i in train_loader:
        train_dic, mask = data_generate(i, 120)
        print(train_dic["input_ids"])
        print(train_dic["position_ids"])
        print(train_dic["token_type_ids"])
        print(train_dic["attention_mask"])
        print(mask)
        break
