import torch
from torch import nn
from transformers import AdamW


if __name__ == '__main__':
    a = torch.randn(256, 12, requires_grad=True)
    b = torch.randn(256, requires_grad=True)
    y = torch.arange(256, dtype=torch.long)
    l = nn.CrossEntropyLoss()
    opt = AdamW([{'params': [a], 'weight_decay': 0.01}, {'params': [b], 'weight_decay': 0}], lr=1e-3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a, b, y = a.to(device), b.to(device), y.to(device)
    for epoch in range(200):
        opt.zero_grad()
        x = a.matmul(a.transpose(1,0))
        x = x + b
        loss = l(x, y)
        print("{}:\t{}".format(epoch, float(loss.cpu())))
        loss.backward()
        opt.step()

