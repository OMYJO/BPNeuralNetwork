from transformers import AdamW
from transformers.modeling_utils import PreTrainedModel
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import os
import src.module.Dataloader as Dataloader


class TrainerV0(nn.Sequential):  # 列表 按顺序放入模块[bert->CNN->relu]->输出self.0
    def fit(self, n_epoch: int, learn_rate: float, train, save_path: str, device="cpu", warm_up: int = 0, dev=None,
            gamma=0.99, step_size=100):
        """

        :param n_epoch:
        :param learn_rate:
        :param train:
        :param save_path:
        :param device:
        :param warm_up:
        :param dev:
        :param gamma:
        :param step_size:
        :return:
        """
        training_loss = []
        optimizer = AdamW(self.parameters(), lr=learn_rate)
        lr_schedule = lambda epoch: 0.1 * (epoch + 1) / warm_up if epoch < warm_up else gamma ** (
                (epoch + 1 - warm_up) // step_size)
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_schedule)
        loss_function = nn.CrossEntropyLoss()
        # datagen = DataGenerate([], device)
        for epoch in range(warm_up + n_epoch):
            self.to(device)  #
            self.train()
            for batch in train:   #
                # batch_tensor, label, mask = datagen(batch)
                train_dic, mask = Dataloader.data_generate(batch)
                optimizer.zero_grad()
                # functor 算子 伪函数
                y_pre = self(train_dic)
                loss = TrainerV0.loss_calculate(y_pre, mask, loss_function=loss_function)
                loss.backward()
                optimizer.step()
                training_loss.append(float(loss.cpu()))
                # torch.cuda.synchronize()
                # train -> make_attention_mask ->attention_mask(替换)->tuple转tensor
                #       -> tuple 转 tensor
                #           [[][][][]] / [[][][][]] / [[][][][]]    (tensor)
                # torch.cuda.synchronize()
            self.eval()
            if dev is not None:  # 胜场预测
                for _ in dev:
                    pass
            self.save(save_path)
            scheduler.step(epoch)
            return training_loss

    @staticmethod
    def make_attention_mask(seq_self, device):    # 我怎么觉得这个函数有点问题
        max_len_q = max(len(seq) for seq in seq_self)
        attention_mask = torch.zeros([len(seq_self), max_len_q, max_len_q], device=device)  # divice
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

    @staticmethod
    def loss_calculate(y_pre, mask, loss_function):
        if len(mask)==0:
            raise ValueError()
        l = [mask[0][2]]
        r = y_pre[mask[0][0], mask[0][1], :].unsqueeze(0)
        for i in range(1, len(mask)):
            l.append(mask[i][2])
            r = torch.cat((r, y_pre[mask[i][0], mask[i][1], :].unsqueeze(0)), 0)
        y_true = torch.Tensor(l)
        return loss_function(r, y_true)

    # x = batch * m(句子长度) * h
    # y = batch * h
    # size(mask_size * n, h) <- x
    # size(mask_size) <- y
    # mask 修改为 一个列表的三元组