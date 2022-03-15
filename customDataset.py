from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, src_path, max_length=None):
        self.data = self.load_data(src_path, max_length)


    def load_data(self, src_path, max_length, encoding='utf-8'):
        data = []
        with open(src_path, encoding=encoding) as f:
            for idx, line in enumerate(f.readlines()):
                if max_length is not None and idx >= max_length:
                    break
                line = line.strip()
                if len(line) > 0:
                    data.append(line)
        return data


if __name__ == '__main__':
    src_path = r'data/PeopleDaily199801.txt'
    dataset = CustomDataset(src_path, max_length=128)
    1
