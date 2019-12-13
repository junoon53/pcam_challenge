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


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    bce = nn.BCELoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float)
            output = model(data)
            test_loss += bce(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


@click.command()
@click.option('--model-type',  default='./saved_models/simpleconv.pt', help='Saved Model Parameters')
@click.option('--model-name',  default='simpleconv', help='Model Type')
def main(model_state, model_name):

    device = torch.device("cuda:1")
    print(torch.cuda.current_device())

    params = { 'batch_size': 512, 'shuffle': True, 'num_workers': 1}


    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[178.6047284, 137.2459255, 176.28579374], std=[59.86620922, 70.70835133, 54.3316497 ]),
    ]) 

    test_dataset = H5Dataset('/home/junoon/Data/PatchCamelyon_v1/test/x.h5', '/home/junoon/Data/PatchCamelyon_v1/test/y.h5', transform)
    test_generator = data.DataLoader(test_dataset, **params)

    model = None
    if model_name == "simpleconv":
        model = simpleconv.SimpleConv()

    model.load_state_dict(torch.load(model_state))
    model.to(device)

    test(None, model, device, test_generator)


if __name__ == "__main__":
    main()

