from transformers import AdamW
from transformers.modeling_utils import PreTrainedModel
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import os
from src.module.Tokenizer import TokenizerV0


class TrainerV0(nn.Sequential):
    def fit(self, n_epoch: int, learn_rate: float, train, save_path: str, device="cpu", warm_up: int = 0, dev=None,
            gamma=0.99, step_size=100):
        optimizer = AdamW(self.parameters(), lr=learn_rate)
        lr_schedule = lambda epoch: 0.1 * (epoch + 1) / warm_up if epoch < warm_up else gamma ** (
                (epoch + 1 - warm_up) // step_size)
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_schedule)
        for epoch in range(warm_up + n_epoch):
            self.to(device)
            self.train()
            for _ in train:
                optimizer.zero_grad()

                optimizer.step()
            self.eval()
            if dev is not None:
                for _ in dev:
                    pass
            self.save(save_path)
            scheduler.step(epoch)

    @staticmethod
    def make_attention_mask(seq_self, device):
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
            module_path = os.path.join(save_directory, "{}_{}".format(i, type(self[i]).__name__))
            os.makedirs(module_path, exist_ok=True)
            if isinstance(self[i], PreTrainedModel):
                self[i].save_pretrained(module_path)
