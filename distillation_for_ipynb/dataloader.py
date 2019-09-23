from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

class MyDataset(Dataset):
    """An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, root=None, train=True, transform=None, 
                 target_transform=None, data_root=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.data_root = data_root

        if self.train:
            self.buf_frame = pd.read_csv(
                self.root + self.data_root)
        else:
            self.buf_frame = pd.read_csv(
                self.root + self.data_root)

    def __getitem__(self, index):
        usr_fea = torch.tensor([self.buf_frame.iloc[index, 1:]])# for i in range(2, self.buf_frame.shape[1])])
        label = torch.tensor(self.buf_frame.iloc[index, 0])

        if self.transform is not None:
            usr_fea = self.transform(usr_fea)

        return (usr_fea, label)

    def __len__(self):
        return len(self.buf_frame)

