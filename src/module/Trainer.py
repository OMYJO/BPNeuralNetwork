from transformers import AdamW
from transformers.modeling_utils import PreTrainedModel
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import os
import module.Dataloader as Dataloader
import json


class TrainerV0(nn.Sequential):  # 列表 按顺序放入模块[bert->CNN->relu]->输出self.0
    def fit(self, n_epoch: int, learn_rate: float, train, save_path: str, device="cpu", warm_up: int = 0, dev=None,
            gamma=0.95, step_size=100, weight_decay: float = 0.01):
        training_loss = []
        best_loss = float("inf")

        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learn_rate)

        lr_schedule = lambda epoch: 0.1 * (epoch + 1) / warm_up if epoch < warm_up else gamma ** (
                (epoch + 1 - warm_up) // step_size)
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_schedule)
        loss_function = nn.CrossEntropyLoss().to(device)
        self.to(device)  #
        # datagen = DataGenerate([], device)
        for epoch in range(warm_up + n_epoch):
            self.train()
            training_loss.append([0, 0])
            for batch in train:   #
                # batch_tensor, label, mask = datagen(batch)
                train_dic, mask = Dataloader.data_generate(batch, device=device)
                optimizer.zero_grad()
                # functor 算子 伪函数
                y_pre = self(train_dic)
                loss = TrainerV0.loss_calculate(y_pre, mask, loss_function=loss_function, device=device)
                loss.backward()
                optimizer.step()
                training_loss[-1][0] += float(loss.cpu()) * len(mask)
                training_loss[-1][1] += len(mask)
                # torch.cuda.synchronize()
                # train -> make_attention_mask ->attention_mask(替换)->tuple转tensor
                #       -> tuple 转 tensor
                #           [[][][][]] / [[][][][]] / [[][][][]]    (tensor)
                # torch.cuda.synchronize()
            training_loss[-1] = training_loss[-1][0] / training_loss[-1][1]
            last_loss = training_loss[-1]
            self.eval()
            if dev is not None:  # 胜场预测
                for _ in dev:
                    pass
            if last_loss < best_loss:
                self.save(save_path)
                best_loss = last_loss
            scheduler.step(epoch)
            with open(os.path.join(save_path, "loss.json"), "w", encoding="utf-8") as f:
                json.dump(training_loss, f)

    @staticmethod
    def make_attention_mask(seq_self, device):    # 我怎么觉得这个函数有点问题
        max_len_q = max(len(seq) for seq in seq_self)
        attention_mask = torch.zeros([len(seq_self), max_len_q, max_len_q], device=device)  # divice
        for i, seq in enumerate(seq_self):
            for j in range(len(seq)):
                for k in range(len(seq)):
                    attention_mask[i, j, k] = 1
        return attention_mask

    @classmethod
    def load(cls, load_path: str, *args):
        assert os.path.isdir(load_path)
        modules = []
        for i, module_class in enumerate(args):
            if issubclass(module_class, PreTrainedModel):
                module_path = os.path.join(load_path, str(i) + "_" + module_class.__name__)
                module = module_class.from_pretrained(module_path)
                modules.append(module)
            else:
                modules.append(module_class())
        model = cls(*modules)
        return model

    def save(self, save_directory: str):
        assert os.path.isdir(save_directory)
        os.makedirs(save_directory, exist_ok=True)
        for i, module in enumerate(self):
            module_path = os.path.join(save_directory, str(i)+"_"+type(module).__name__)
            os.makedirs(module_path, exist_ok=True)
            if isinstance(module, PreTrainedModel):
                module.save_pretrained(module_path)

    @staticmethod
    def loss_calculate(y_pre, mask, loss_function, device):
        if len(mask)==0:
            raise ValueError()
        l = [mask[0][2]]
        r = y_pre[mask[0][0], mask[0][1], :].unsqueeze(0)
        for i in range(1, len(mask)):
            l.append(mask[i][2])
            r = torch.cat((r, y_pre[mask[i][0], mask[i][1], :].unsqueeze(0)), 0)
        y_true = torch.tensor(l, device=device)
        return loss_function(r, y_true)

    # x = batch * m(句子长度) * h
    # y = batch * h
    # size(mask_size * n, h) <- x
    # size(mask_size) <- y
    # mask 修改为 一个列表的三元组