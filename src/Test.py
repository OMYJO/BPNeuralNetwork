import os
import json
from torch.utils.data import dataloader
from transformers import BertConfig
from src.module.BERT import BertV0
from src.module.Pooling import SequencePoolingV0, MLMPoolingV0
from src.module.Trainer import TrainerV0
from src.module.Tokenizer import TokenizerV0
from src.module.Tokenizer import TokenizerV1
from src.dataset.DataSet import ListDataSetV0
from src.dataset.DataSet import LabelDataSetV0

def main1():
    config = BertConfig(hidden_size=16, max_position_embeddings=32, type_vocab_size=8, vocab_size=256,
                        num_attention_heads=4, intermediate_size=32)
    trainer = TrainerV0(
        BertV0(config),
        SequencePoolingV0(),
        MLMPoolingV0(config)
    )
    trainer.save_pretrained(os.path.join("..", "models", "version0"))


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
    train_set = LabelDataSetV0(data_set, label_set)
    train_loader = dataloader.DataLoader(train_set, batch_size=16, shuffle=False,
                                         collate_fn=lambda u: ([i[0] for i in u], [i[1] for i in u]))
    for x, y in train_loader:
        print(x, y)
        break


if __name__ == '__main__':
    main3()
