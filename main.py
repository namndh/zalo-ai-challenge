import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from dataset import HitSongDataset
from utils import RMSELoss


torch.manual_seed(42)

net = torch.nn.Sequential(
    torch.nn.Linear(9, 200),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(200, 100),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(100, 1)
)

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
criterion = RMSELoss()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

batch_size = 16
epochs = 250
save_training_loss = 0
save_eval_acc = 0


train_dataset = HitSongDataset('', '', '')
test_dataset = HitSongDataset('', '', '', train=False)

train_loader = Data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
test_loader = Data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)

net.train()
for epoch in range(epochs):
    training_loss = 0
    for idx, (datas, labels) in enumerate(train_loader):
        x = datas.to(device)
        y = labels.to(device)
        predictions = net(x)

        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        if idx % 1000 == 999:    # print every 1000 mini-batches
            print('[%d, %5d] Training loss: %.3f' %
                  (epoch + 1, idx + 1, training_loss / 2000))
    if training_loss < save_training_loss:


# Eva