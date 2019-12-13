import time
import click
import torch
from torch.utils import data
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from dataset import H5Dataset
from models import simpleconv
from test import test

def train(log_interval, model, device, train_generator, optimizer, epoch):
    model.train()
    bce = nn.BCELoss()
    for batch_idx, (data, target) in enumerate(train_generator):
        data, target = data, target.squeeze()
        # print(data[0])
        data, target = data.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float)
        # print(type(data), data.shape, type(target), target.shape)
        optimizer.zero_grad()
        output = model(data)
        # print(output, target)
        # print('==========')
        loss = bce(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_generator.dataset),
                100. * batch_idx / len(train_generator), loss.item()))

@click.command()
@click.option('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 64)')
@click.option('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
@click.option('--lr', type=float, default=0.0001,  metavar='LR',
                    help='learning rate (default: 0.001)')
@click.option('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
@click.option('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
@click.option('--log-interval', type=int, default=30, metavar='N',
                    help='how many batches to wait before logging training status')
@click.option('--save-model',  default=True, help='For Saving the current Model')
def main(batch_size, epochs, lr, momentum, seed, log_interval, save_model):
    # Training settings
    torch.manual_seed(seed)

    device = torch.device("cuda:1")
    print(torch.cuda.current_device())

    params = { 'batch_size': batch_size, 'shuffle': True, 'num_workers': 1}


    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[178.6047284, 137.2459255, 176.28579374], std=[59.86620922, 70.70835133, 54.3316497 ]),
    ]) 

    train_dataset = H5Dataset('/home/junoon/Data/PatchCamelyon_v1/train/x.h5', '/home/junoon/Data/PatchCamelyon_v1/train/y.h5', transform)
    train_generator = data.DataLoader(train_dataset, **params)

    valid_dataset = H5Dataset('/home/junoon/Data/PatchCamelyon_v1/valid/x.h5', '/home/junoon/Data/PatchCamelyon_v1/valid/y.h5', transform)
    valid_generator = data.DataLoader(valid_dataset, **params)

    
    model = simpleconv.SimpleConv().to(device)
    print("learning rate", lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train(log_interval, model, device, train_generator, optimizer, epoch)
        test(None, model, device, valid_generator)


    print(f"""Total time taken: {time.time() - start_time}""")

    if save_model:
        torch.save(model.state_dict(), "./saved_models/simpleconv.pt")



if __name__ == "__main__":
    main()

