from myModel import *
from customDataset import *
from tqdm import tqdm, trange
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import time
from myPlot import MyPlot

src_path = './data/PeopleDaily199801.txt'

epoches = 18 + 1
test_ratio = 0.2
batch_size = 64
embedding_dim = 128
hidden_dim = 32
save_path = './checkpoint/myModel.ckpt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('current device:', device)

if __name__ == '__main__':
    dataset = CustomDataset(src_path, max_length=1024)
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset =\
        random_split(
            dataset,(train_size, test_size)
        )
    train_dataLoader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataLoader = DataLoader(test_dataset, batch_size=batch_size)
    model = BiLSTM_CRF(dataset.vocab_size, dataset.tag2idx, embedding_dim, hidden_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epoches):
        mean_loss = model.train_one_epoch(train_dataLoader, optimizer, batch_size)
        print(f'epoch: {epoch} \t train_loss: {mean_loss}')
        if epoch % 3 == 0:
            mean_loss, accuracy = model.test(test_dataLoader, batch_size)
            print(f'val_loss: {mean_loss} \t val_accuracy: {accuracy}')
        if (epoch + 1) % 5 == 0:
            model.auto_save(save_path)
    # t = time.localtime()
    # month, day, hour, minute = t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min
    # model.save_model(save_path + f'{month}-{day} {hour}:{minute}')
    # MyPlot.plot_curve(range(epoches), train_losses, l='train_loss')
    # MyPlot.plot_curve(range(len(test_losses)), test_losses, l='test_losses')











