from transformers import AdamW
from transformers.modeling_utils import PreTrainedModel
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import os
from src.module.Pooling import MLMPoolingV0
from src.module.Tokenizer import TokenizerV0
import random
import copy
import json


class TrainerV0(nn.Sequential):  # 列表 按顺序放入模块[bert->CNN->relu]->输出self.0
    def fit(self, n_epoch: int, learn_rate: float, train, save_path: str, device="cpu", warm_up: int = 0, dev=None,
            gamma=0.99, step_size=100):
        optimizer = AdamW(self.parameters(), lr=learn_rate)
        lr_schedule = lambda epoch: 0.1 * (epoch + 1) / warm_up if epoch < warm_up else gamma ** (
                (epoch + 1 - warm_up) // step_size)
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_schedule)
        input, ground_truth, mask_position = tokenizer_train(train)
        model = PreTrainedModel()
        loss = nn.CrossEntropyLoss()
        for epoch in range(warm_up + n_epoch):
            self.to(device)
            self.train()
            for i in range(len(input)):   #
                predict = model.forward(input)
                loss = loss(predict, target.long())
                optimizer.zero_grad()
                # torch.cuda.synchronize()
                loss.backward()
                # train -> make_attention_mask ->attention_mask(替换)->tuple转tensor
                #       -> tuple 转 tensor
                #           [[][][][]] / [[][][][]] / [[][][][]]    (tensor)
                # torch.cuda.synchronize()
                optimizer.step()
            self.eval()
            if dev is not None:  # 胜场预测
                for _ in dev:
                    pass
            self.save(save_path)
            scheduler.step(epoch)

    @staticmethod
    def make_attention_mask(seq_self, device):    # 我怎么觉得这个函数有点问题
        max_len_q = max(len(seq) for seq in seq_self)
        attention_mask = torch.zeros([len(seq_self), max_len_q, max_len_q], device=device)
        for i, seq in enumerate(seq_self):
            for j in range(len(seq)):
                for k in range(len(seq)):
                    attention_mask[i, j, k] = 1
        return attention_mask

    def save_pretrained(self, save_directory: str):
        assert os.path.isdir(save_directory)
        os.makedirs(save_directory, exist_ok=True)
        for i in range(len(self)):
            module_path = os.path.join(save_directory, str(i))
            os.makedirs(module_path, exist_ok=True)
            if isinstance(self[i], PreTrainedModel):
                self[i].save_pretrained(module_path)

    def tokenizer_path(self, data_name):
        data_name = str(data_name)+".json"
        vocab = os.path.join("../../", "models", "version0", "vocab.txt")
        tokenizer = TokenizerV0(64, vocab, "[CLS]", "[SEP]")
        dic_len = len(tokenizer.vocab)
        with open(os.path.join("../../", "data", data_name), "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = os.path.join("../../", "models", "version0", "vocab.txt")
        tokenizer = TokenizerV0(64, vocab, "[CLS]", "[SEP]")
        token = []
        for match in data:
            token.append(tokenizer.tokenize(match, match[0]["is_overallBP"], True))
        token, position, ground_truth, type, mask_position = tensor_generate(token, dic_len)
        return input, ground_truth, mask_position

    def tokenizer_train(self, train):
        vocab = os.path.join("../../", "models", "version0", "vocab.txt")
        tokenizer = TokenizerV0(64, vocab, "[CLS]", "[SEP]")
        token = []
        dic_len = len(tokenizer.vocab)
        for match in train:
            token.append(tokenizer.tokenize(match, match[0]["is_overallBP"], True))
        token, position, ground_truth, type, mask_position = tensor_generate(token, dic_len)
        return input, ground_truth, mask_position

    def zero_file(self, short_list, max_len):
        for i in range(max_len - len(short_list)):
            short_list.append(0)
        return short_list

    def tensor_generate(self, train, dic_len):
        token = []
        type = []
        position = []
        ground_truth = []
        mask_position = []
        attention_mask = make_attention_mask(train)  # 我怎么觉得这个函数有点问题
        mask_rule = lambda rand: 3 if rand < 0.8 else (battle_copy[hero] if rand > 0.9 else random.randint(16, dic_len))
        max_len = 15
        for competition in train:
            for battle in competition:
                if len(battle[0])<max_len:
                    max_len = len(battle[0])
        for competition in train:
            for battle in competition:
                ground_truth.append(zero_file(battle[0], max_len))
                position.append(zero_file(battle[1], max_len))
                type.append(zero_file(battle[2]), max_len)
                battle_copy = copy.deepcopy(battle[0])
                battle_mask = [0 for i in range(max_len)]

                for hero in range(len(battle_copy)):
                    if random.random() < 0.15:
                        battle_mask[hero] = 1
                        masktype = random.random()
                        battle_copy[hero] = mask_rule(masktype)
                mask_position.append(zero_file(battle_mask))
                token.append(zero_file(battle_copy), max_len)
        token = torch.Tensor(token)
        ground_truth = torch.Tensor(ground_truth)
        position = torch.Tensor(position)
        type = torch.Tensor(type)
        input = []
        for i in range(len(token)):
            input_single = {}
            input_single["attention_mask"] = attention_mask[i]
            input_single["input_ids"] = token[i]
            input_single["position_ids"] = position[i]
            input_single["token_type_ids"] = type[i]
            input.append(input_single)
        return input, ground_truth, mask_position






