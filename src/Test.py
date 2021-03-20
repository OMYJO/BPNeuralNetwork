import os
import json
import datetime
import torch
from torch.utils.data import dataloader
from transformers import BertConfig
from module.BERT import BertV0
from module.Pooling import SequencePoolingV0, MLMPoolingV0
from module.Trainer import TrainerV0
from module.Tokenizer import TokenizerV0
from module.Tokenizer import TokenizerV1
from dataset.DataSet import ListDataSetV0
from dataset.DataSet import MultiListDataSetV0


def main1():
    config = BertConfig(hidden_size=16, max_position_embeddings=32, type_vocab_size=8, vocab_size=256,
                        num_attention_heads=4, intermediate_size=32)
    trainer = TrainerV0(
        BertV0(config),
        SequencePoolingV0(),
        MLMPoolingV0(config)
    )
    trainer.save(os.path.join("..", "models", "version0"))


def main2():
    vocab = os.path.join("../", "models", "version0", "vocab.txt")
    tokenizer = TokenizerV0(64, vocab, "[CLS]", "[SEP]")
    data_set = []
    basic_path = os.path.join("..", "data")
    for file_name in os.listdir(basic_path):
        if "json" in file_name:
            with open(os.path.join(basic_path, file_name), "r", encoding="utf-8") as f:
                matchs = list(json.load(f))
                for match in matchs:
                    data_set += tokenizer.tokenize(match, match[0]["is_overallBP"], True)
    train_set = ListDataSetV0(data_set)
    train_loader = dataloader.DataLoader(train_set, batch_size=16, shuffle=False, collate_fn=lambda u: u)
    for x in train_loader:
        print(x)
        break


def main3():
    vocab = os.path.join("../", "models", "version0", "vocab.txt")
    tokenizer = TokenizerV1(64, vocab, "[CLS]", "[SEP]")
    data_set = []
    label_set = []
    basic_path = os.path.join("..", "data")
    for file_name in os.listdir(basic_path):
        if "json" in file_name:
            with open(os.path.join(basic_path, file_name), "r", encoding="utf-8") as f:
                matchs = list(json.load(f))
                for match in matchs:
                    x, y = tokenizer.tokenize(match)
                    data_set += x
                    label_set += y
    train_set = MultiListDataSetV0(data_set, label_set)
    train_loader = dataloader.DataLoader(train_set, batch_size=16, shuffle=False,
                                         collate_fn=lambda u: ([i[0] for i in u], [i[1] for i in u]))
    for x, y in train_loader:
        print(x, y)
        break


def main4():
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    # epochs = 50  # 235  # 65  # 100  # 200
    # lr = 1e-3  # 5e-3  # 5e-4  # 5e-4  # 2e-5
    # warm_up = 0  # 20
    # batch_size = 8192
    # epochs = 50  # 200
    # lr = 1e-4
    # warm_up = 0
    # batch_size = 2048
    # epochs = 100  # 200 # 500 # 50
    # lr = 2e-3  # 1e-2 # 1e-3
    # warm_up = 10  # 5
    # batch_size = 2048
    # epochs = 500
    # lr = 1e-2
    # warm_up = 20
    # batch_size = 2048
    epochs = 500
    lr = 1e-2
    warm_up = 20
    batch_size = 2048
    # bert_config = BertConfig(vocab_size=256, hidden_size=12, num_hidden_layers=1, num_attention_heads=3,
    #                          intermediate_size=48, max_position_embeddings=80, type_vocab_size=8)
    # bert_config = BertConfig(vocab_size=121, hidden_size=12, num_hidden_layers=1, num_attention_heads=3,
    #                          intermediate_size=48, max_position_embeddings=80, type_vocab_size=8)
    # bert_config = BertConfig(vocab_size=121, hidden_size=12, num_hidden_layers=4, num_attention_heads=3,
    #                          intermediate_size=48, max_position_embeddings=80, type_vocab_size=8)
    bert_config = BertConfig(vocab_size=121, hidden_size=12, num_hidden_layers=1, num_attention_heads=3,
                             intermediate_size=48, max_position_embeddings=80, type_vocab_size=8)
    trainer = TrainerV0(BertV0(bert_config), SequencePoolingV0(), MLMPoolingV0(bert_config))
    # trainer = TrainerV0.load("../models/version0/20210313220450", BertV0, SequencePoolingV0, MLMPoolingV0)
    # trainer = TrainerV0.load("../models/version0/20210315145313", BertV0, SequencePoolingV0, MLMPoolingV0)
    # trainer = TrainerV0.load("../models/version0/20210315161752", BertV0, SequencePoolingV0, MLMPoolingV0)

    def filter_(parameters):
        r = []
        for k, v in parameters:
            if any(k.startswith(s) for s in ["0.encoder.layer.1", "2."]):
                r.append((k, v))
        return r

    now = datetime.datetime.today()
    save = os.path.join("..", "models", "version0", now.strftime("%Y%m%d%H%M%S"))
    os.makedirs(save, exist_ok=True)
    vocab = os.path.join("..", "models", "version0", "vocab.txt")
    tokenizer = TokenizerV0(80, vocab, "[CLS]", "[SEP]")
    data_set = []
    basic_path = os.path.join("..", "data")
    for file_name in os.listdir(basic_path):
        if "json" in file_name:
            with open(os.path.join(basic_path, file_name), "r", encoding="utf-8") as f:
                matchs = list(json.load(f))
                for match in matchs:
                    data_set += tokenizer.tokenize(match, match[0]["is_overallBP"], True)
    train_set = ListDataSetV0(data_set)
    train_loader = dataloader.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=lambda u: u)
    trainer.fit(learn_rate=lr,
                n_epoch=epochs,
                train=train_loader,
                save_path=save,
                device=device,
                warm_up=warm_up)
                #_filter=filter_)


def deeper():
    new_state_dict = {}
    with open("../models/version0/20210313190112/0_BertV0/pytorch_model.bin", "rb") as f:
        old_state_dict = torch.load(f, map_location="cpu")
        for k, v in old_state_dict.items():
            new_state_dict[k] = v
            if "encoder.layer.0" in k:
                new_state_dict[k.replace("encoder.layer.0", "encoder.layer.2")] = v.clone()
            elif "encoder.layer.1" in k:
                new_state_dict[k.replace("encoder.layer.1", "encoder.layer.3")] = v.clone()
    with open("pytorch_model.bin", "wb") as f1:
        torch.save(new_state_dict, f1)


def main5():
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    epochs = 500
    lr = 1e-2
    warm_up = 20
    batch_size = 2048
    bert_config = BertConfig(vocab_size=121, hidden_size=12, num_hidden_layers=1, num_attention_heads=6,
                             intermediate_size=48, max_position_embeddings=80, type_vocab_size=8)
    trainer = TrainerV0(BertV0(bert_config), SequencePoolingV0(), MLMPoolingV0(bert_config))

    def filter_(parameters):
        r = []
        for k, v in parameters:
            if any(k.startswith(s) for s in ["0.encoder.layer.1", "2."]):
                r.append((k, v))
        return r

    now = datetime.datetime.today()
    save = os.path.join("..", "models", "version0", now.strftime("%Y%m%d%H%M%S"))
    os.makedirs(save, exist_ok=True)
    vocab = os.path.join("..", "models", "version0", "vocab.txt")
    tokenizer = TokenizerV0(80, vocab, "[CLS]", "[SEP]")
    data_set = []
    basic_path = os.path.join("..", "data")
    for file_name in os.listdir(basic_path):
        if "json" in file_name:
            with open(os.path.join(basic_path, file_name), "r", encoding="utf-8") as f:
                matchs = list(json.load(f))
                for match in matchs:
                    data_set += tokenizer.tokenize(match, match[0]["is_overallBP"], True)
    train_set = ListDataSetV0(data_set)
    train_loader = dataloader.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=lambda u: u)
    dev_set0 = []
    dev_set1 = []
    for i, x in enumerate(data_set):
        for j, typ in enumerate(x[2]):
            if (typ == 0 or typ == 1) and x[0][j] >= 16:
                ids = x[0].copy()
                ids[j] = 3
                dev_set0.append([ids, x[1], x[2]])
                dev_set1.append([j, x[0][j]])
    dev_set = MultiListDataSetV0(dev_set0, dev_set1)
    dev_loader = dataloader.DataLoader(dev_set, batch_size=batch_size, shuffle=True, collate_fn=lambda u: u)
    trainer.fit(learn_rate=lr,
                n_epoch=epochs,
                train=train_loader,
                dev=dev_loader,
                save_path=save,
                device=device,
                warm_up=warm_up)
                # _filter=filter_)



if __name__ == '__main__':
    # main3()
    main4()
    # deeper()