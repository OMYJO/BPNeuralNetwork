from transformers import AdamW
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from src.module.Tokenizer import TokenizerV0

class TrainerV0(nn.Sequential):

    def fit(self, n_epoch: int, learn_rate: float, warm_up: int, train, dev=None, gamma=0.99, step_size=100):
        optimizer = AdamW(self.parameters(), lr=learn_rate)
        lr_schedule = lambda epoch: 0.1 * (epoch + 1) / warm_up if epoch < warm_up else gamma ** (
                (epoch + 1 - warm_up) // step_size)
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_schedule)

    def make_attention_mask(self, seq_s, device):
        max_len_q = max(len(seq) for seq in seq_s)
        attention_mask = torch.zeros([len(seq_s), max_len_q, max_len_q], device=device)
        for i, seq in enumerate(seq_s):
            for j in range(len(seq)):
                for k in range(len(seq)):
                    attention_mask[i, j, k] = 1
        return attention_mask
