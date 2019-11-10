import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from dataset import HitSongDataset
from utils import RMSELoss
from tqdm import tqdm

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
tolerance = 0.1


train_dataset = HitSongDataset('', '', '')
val_dataset = HitSongDataset('', '', '', train=False)

train_loader = Data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
val_loader = Data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)

for epoch in tqdm(range(epochs)):
    training_loss = 0
    net.train()
    for idx, (datas, labels) in tqdm(enumerate(train_loader)):
        print('Training...')
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
        print('Saving model at {}-th epoch with loss is {}'.format(epoch+1, training_loss))
        ckpt = {'epoch':epoch,
                'lowest_loss_train':training_loss,
                'model':net.module.state_dict(),
                'optimizer': optimizer.state_dict()}
        torch.save(ckpt, 'model/train_best.pt')

    val_acc = 0
    n_corrects = 0
    net.eval()
    with torch.no_grad():
        for idx, (datas, labels) in tqdm(enumerate(val_loader)):
            print('Validating...')
            datas, labels = datas.to(device), labels.to(device)
            outputs = net(datas)
            predictions = outputs.view(len(labels))

            n_correct = torch.sum(torch.abs(predictions-labels) < tolerance)
            n_corrects += n_correct

    val_acc = n_corrects*100/len(val_dataset)
    print('Validation accuracy is: {}%'.format(val_acc))
    if save_eval_acc < val_acc:
        print('Saving model at {}-th epoch with val acc is {}'.format(epoch+1, val_acc))
        ckpt = {'epoch':epoch,
                'lowest_loss_train':training_loss,
                'model':net.module.state_dict(),
                'optimizer': optimizer.state_dict()}
        torch.save(ckpt, 'model/val_best.pt')
