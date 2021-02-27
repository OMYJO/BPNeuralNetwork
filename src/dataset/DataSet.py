from torch.utils.data.dataset import Dataset


class ListDataSetV0(Dataset):

    def __init__(self, dataset) -> None:
        super().__init__()
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, item):
        return self._dataset[item]


class LabelDataSetV0(Dataset):
    def __init__(self, *dataset):
        self._dataset = dataset

    def __getitem__(self, index: int):
        return tuple(dataset[index] for dataset in self._dataset)

    def __len__(self) -> int:
        return len(self._dataset[0])
