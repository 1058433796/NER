import numpy as np
from torch.utils.data import Dataset
from helper import Helper


class CustomDataset(Dataset):
    def __init__(self, src_path, max_length=None):
        self.helper = Helper()
        data = self.load_data(src_path, max_length, 23)
        # X: (w1, w2, w3) y: (y1, y2, y3)
        X, y = self.helper.transform(data)
        # 生成映射字典
        self.w2idx, self.idx2w = self.helper.get_mapper(X, params=['<PAD>'])
        self.tag2idx, self.idx2tag = self.helper.get_mapper(y, params=['<PAD>'])

        tokened_X = self.helper.tokenize(X, self.w2idx)
        tokened_y = self.helper.tokenize(y, self.tag2idx)
        del X, y
        self.X = self.helper.pad_tokens(tokened_X, self.w2idx['<PAD>'], batch_first=True)
        self.y = self.helper.pad_tokens(tokened_y, self.tag2idx['<PAD>'], batch_first=True)
        del tokened_X, tokened_y
        self.mask = self.helper.get_mask(self.X, self.w2idx['<PAD>'])
        self.vocab_size = self.helper.get_vocab_size(self.X)

    def __getitem__(self, item):
        X, y, mask = self.X[item], self.y[item], self.mask[item]
        return X,y, mask

    def __len__(self):
        return len(self.X)

    def load_data(self, src_path, max_length, offset=None, encoding='utf-8'):
        data = []
        with open(src_path, encoding=encoding) as f:
            for idx, line in enumerate(f.readlines()):
                if max_length is not None and idx >= max_length:
                    break
                line = line.strip()[offset:]
                if len(line) > 0:
                    data.append(line)
        return data


if __name__ == '__main__':
    src_path = r'data/PeopleDaily199801.txt'
    dataset = CustomDataset(src_path, max_length=24)
    print(len(dataset[0][0]), len(dataset[0][1]))
    print(len(dataset[1][0]), len(dataset[1][1]))
