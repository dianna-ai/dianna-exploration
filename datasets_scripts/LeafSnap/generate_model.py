#!/usr/bin/env python
# coding: utf-8

import os
import tqdm
import argparse

import wandb
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms


class Network(nn.Module):
    def __init__(self, img_size, num_classes, kernel_size, pooling_size, num_channels):
        super().__init__()

        self.num = len(num_channels)

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        for i, out_channels in enumerate(num_channels):
            if i == 0:
                in_channels = 3
            else:
                in_channels = num_channels[i - 1]
            padding = int((kernel_size - 1) / 2)
            self.conv_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding))
            self.bn_layers.append(nn.BatchNorm2d(out_channels))
            # no pooling layer for the last step
            if i < self.num - 1:
                self.pool_layers.append(nn.MaxPool2d(pooling_size, pooling_size))

        # image size has been reduced by a factor pooling_size as many times as there are pooling layres
        reduced_img_size = int(img_size / pooling_size ** len(self.pool_layers))
        self.final_layer_size = num_channels[-1] * reduced_img_size ** 2

        self.fc1 = nn.Linear(self.final_layer_size, num_classes)

    def forward(self, data):
        for i in range(self.num):
            conv = self.conv_layers[i]
            bn = self.bn_layers[i]
            # apply pooling, but not in last step
            if i < self.num - 1:
                pool = self.pool_layers[i]
                data = pool(F.relu(bn(conv(data))))
            else:
                data = F.relu(bn(conv(data)))

        data = data.view(-1, self.final_layer_size)
        data = self.fc1(data)
        return data


def accuracy(model_output, y_true):
    _, y_pred = torch.max(model_output, dim=1)
    return (y_pred == y_true).sum() / len(y_pred)


def train(model, train_data, optimizer, loss_func):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in tqdm.tqdm(train_data):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)

        loss = loss_func(predictions, labels)
        acc = accuracy(predictions, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        wandb.log({'train_loss': loss.item(), 'train_acc': acc.item()})

    epoch_loss /= len(train_data)
    epoch_acc /= len(train_data)

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, data, loss_func, train_mode=False):
    avg_loss = 0
    avg_acc = 0
    if train_mode:
        model.train()
    else:
        model.eval()

    for batch in data:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)

        loss = loss_func(predictions, labels)
        acc = accuracy(predictions, labels)

        avg_loss += loss.item()
        avg_acc += acc.item()

    avg_loss /= len(data)
    avg_acc /= len(data)

    return avg_loss, avg_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--pooling', type=int, default=2)
    parser.add_argument('--channels1', type=int, default=12)
    parser.add_argument('--channels2', type=int, default=12)
    parser.add_argument('--channels3', type=int, default=24)
    parser.add_argument('--channels4', type=int, default=24)
    parser.add_argument('--nowandb', action='store_true')

    args = parser.parse_args()
    assert args.kernel_size % 2 == 1, "kernel size must be odd to keep same image dimensions after conv"

    config = wandb.config
    for key, value in vars(args).items():
        setattr(config, key, value)

    if args.nowandb:
        mode = 'disabled'
    else:
        mode = 'online'

    wandb.init(project='leafsnap', entity='dianna-ai', mode=mode)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset_root = '../../../leafsnap-dataset-30subset'

    transform = transforms.Compose([transforms.Resize(config.img_size),
                                    transforms.CenterCrop(config.img_size),
                                    transforms.ToTensor()])
    images_root = os.path.join(dataset_root, 'dataset/images_to_use_bmp')  # loading bmp images is about twice as fast as jpg
    dataset = datasets.ImageFolder(images_root, transform=transform)

    val_frac = .1
    test_frac = .1

    num_samples = len(dataset)
    num_classes = len(dataset.classes)

    val_samples = int(val_frac * num_samples)
    test_samples = int(test_frac * num_samples)
    train_samples = num_samples - val_samples - test_samples

    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_samples, val_samples, test_samples], generator=torch.Generator().manual_seed(42))

    workers = 12

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    # don't need the test data in a wandb sweep
    # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    channels = [config.channels1, config.channels2, config.channels3, config.channels4]
    model = Network(config.img_size, num_classes, config.kernel_size, config.pooling, channels).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_func = nn.CrossEntropyLoss().to(device)

    for epoch in range(config.epochs):
        # run training loop
        train_loss, train_acc = train(model, train_dataloader, optimizer, loss_func)
        # evalute on validation data, with same model settings as during training (i.e. including dropout)
        val_loss, val_acc = evaluate(model, val_dataloader, loss_func, train_mode=True)

        wandb.log({'train_epoch_loss': train_loss, 'train_epoch_acc': train_acc})
        wandb.log({'val_epoch_loss': val_loss, 'val_epoch_acc': val_acc})
