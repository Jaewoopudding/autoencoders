import torch
import os
from torchvision import datasets, transforms


def data_download():
    train_data = datasets.MNIST(root='../dataset',
                                train=True,
                                download=False,
                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]))
    test_data = datasets.MNIST(root='../dataset',
                               train=False,
                               download=False,
                               transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]))
    print('number of training data : ', len(train_data))
    print('number of test data : ', len(test_data))
    return train_data, test_data


def data_loader(train_data, test_data, batch_size):
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size,
                                              shuffle=True)
    return train_loader, test_loader


if __name__ == '__main__':
    print(os.getcwd())
    data_download()
