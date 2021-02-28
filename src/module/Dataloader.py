import os
import random
import json
from torch.utils.data import dataloader
from transformers import BertConfig
from src.module.BERT import BertV0
from src.module.Pooling import SequencePoolingV0, MLMPoolingV0
from src.module.Trainer import TrainerV0
from src.module.Tokenizer import TokenizerV0
from src.dataset.DataSet import ListDataSetV0

class DataGenerate():
    def __init__(self):
        self.vocab_len = 100
        self.train_data = []
        self.train_mask = []
        self.ground_truth = []
        self.train_data_tensor = []
        # self.train_loader = dataloader.DataLoader()
        self.Dataloader()
        self.make_mask()

    def Dataloader(self):
        vocab = os.path.join("../", "..", "models", "version0", "vocab.txt")
        tokenizer = TokenizerV0(64, vocab, "[CLS]", "[SEP]")
        self.vocab_len = len(tokenizer.vocab)
        data_set = []
        basic_path = os.path.join("..", "..", "data")
        for file_name in os.listdir(basic_path):
            if "json" in file_name:
                with open(os.path.join(basic_path, file_name), "r", encoding="utf-8") as f:
                    matchs = list(json.load(f))
                    for match in matchs:
                        data_set += tokenizer.tokenize(match, match[0]["is_overallBP"], True)
        train_set = ListDataSetV0(data_set)
        self.train_loader = dataloader.DataLoader(train_set, batch_size=16, shuffle=False, collate_fn=lambda x:x)

    def make_mask(self):
        mask_rule = lambda rand: 3 if rand < 0.8 else (competition[0][hero_id] if rand > 0.9 else random.randint(16, self.vocab_len))
        for data_batch in self.train_loader:
            batch = []
            mask_position = []
            ground_truth = []
            for competition in data_batch:
                hero = []
                mask = []
                for hero_id in range(len(competition[0])):
                    if (competition[2][hero_id] == 1)or(competition[2][hero_id] == 2)or(competition[2][hero_id] == 3)or(competition[2][hero_id] == 4):
                        if random.random() < 0.15:
                            mask.append(1)
                            hero.append(mask_rule(random.random()))
                        else:
                            hero.append(competition[0][hero_id])
                            mask.append(0)
                    else:
                        hero.append(competition[0][hero_id])
                        mask.append(0)
                battle = (hero, competition[1], competition[2])
                batch.append(battle)
                ground_truth.append(competition[0])
                mask_position.append(mask)
            self.train_data.append(batch)
            self.train_mask.append(mask_position)
            self.ground_truth.append(ground_truth)

    def TupleDataToTensor(self):
        for tuple_data in self.train_data:
            train_cache = []
            for i in tuple_data:
                train_cache.append(torch.Tensor(i))
            self.train_data_tensor.append(train_cache)

    def get_train_data(self):
        return self.train_data_tensor

    def get_train_mask(self):
        return self.train_mask

    def get_ground_truth(self):
        return self.ground_truth


if __name__=='__main__':
    dataloader = DataGenerate()
    print(len(dataloader.train_data))
    print(len(dataloader.train_mask))
    print(len(dataloader.ground_truth))
