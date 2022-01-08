# coding=utf-8
import os

from dataset import SynthesisDataset
from model import ConfidenceControl

import argparse

import numpy as np
import pandas as pd
import torch
from thop import profile, clever_format
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

import warnings

warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

import torch.nn.functional as F
# from model import ConfidenceControl, ConvAngularPenCC
from utils import MPerClassSampler
from torch.distributions import normal
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import phate

import io
import matplotlib.pyplot as plt
import PIL.Image
import seaborn as sns
from torchvision.transforms import ToTensor
import colorcet as cc

from sklearn.metrics.pairwise import pairwise_distances


def setCuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    if device.type != 'cpu':
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())
        device_count = torch.cuda.device_count()
    else:
        device_count = 1

    return device, device_count


def setSeed():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parseArgument():
    parser = argparse.ArgumentParser(description='Train Model')

    parser.add_argument('--data_path', default='/data/hbyoo', type=str, help='datasets path')
    parser.add_argument('--data_name', default='mnist', type=str,
                        choices=['car', 'cub', 'sop', 'isc', 'mnist'],
                        help='dataset name')
    parser.add_argument('--crop_type', default='uncropped', type=str, choices=['uncropped', 'cropped'],
                        help='crop data or not, it only works for car or cub dataset')

    parser.add_argument('--batch_size', default=512, type=int, help='train batch size')
    parser.add_argument('--num_sample', default=4, type=int, help='samples within each class')
    parser.add_argument('--feature_dim', default=2, type=int, help='feature dim')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='learning rate scheduler gamma')
    parser.add_argument('--num_epochs', default=30, type=int, help='train epoch number')

    opt = parser.parse_args()

    return opt.data_path, opt.data_name, opt.crop_type, opt.batch_size, opt.num_sample, opt.feature_dim, opt.lr, opt.lr_gamma, opt.num_epochs


def loadData(type):
    train_dataset = SynthesisDataset(train=True, data_type=type)
    train_data_loader = DataLoader(train_dataset, batch_size=1600, shuffle=True)
    test_dataset = SynthesisDataset(train=False, data_type=type)
    test_data_loader = DataLoader(test_dataset, batch_size=1600, shuffle=True)
    return train_data_loader, test_data_loader, 2


def setModel(device, feature_dim, number_of_class, model_type):
    model = ConfidenceControl(feature_dim, number_of_class, model_type)
    # writer.add_graph(model)
    return model.to(device)


def setOptimizer(device, model, lr):
    # 첫번째 에폭에는 feature extractor의 가중치는 변경시키고 싶지 않음, 피쳐 추출기의 가중치는 완벽하니까, 분류기가 그에 맞추도록
    optimizer_init = SGD(
        [{'params': model.feature_extractor.refactor.parameters()}, {'params': model.classifier.parameters()}],
        lr=lr, momentum=0.9, weight_decay=1e-4)
    # 두번째 에폭에는 분류기가 어느정도 안정되었을 것이므로, 피쳐 추출기도 같이 학습시키자.
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    return optimizer, optimizer_init


def accuracy(net, device, dataloader):
    net.eval()
    correct = 0
    total = len(dataloader.dataset)
    for inputs, labels in tqdm(dataloader, dynamic_ncols=True):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = net.forward(inputs)
            pred = outputs.argmax(dim=1)
            correct += torch.eq(pred, labels.reshape(-1)).sum().float().item()

    return correct, total


def train(net, train_data_loader, device, criterion, optimizer):
    total_loss = 0
    iternum = 0
    for inputs, labels in tqdm(train_data_loader, dynamic_ncols=True):
        inputs, labels = inputs.to(device), labels.reshape(-1).to(device)

        optimizer.zero_grad()
        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        iternum += 1

    average_loss = total_loss / iternum
    return average_loss


def get_embed(net, device, dataloader, feature_dim):
    embedding_list = np.zeros(shape=(0, feature_dim))
    label_list = np.zeros(shape=(0))
    for inputs, labels in tqdm(dataloader, dynamic_ncols=True):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = net.forward_feature(inputs)
        label_list = np.concatenate((label_list, labels.detach().cpu().numpy().ravel()))
        embedding_list = np.concatenate([embedding_list, outputs.detach().cpu().numpy()], axis=0)

    return embedding_list, label_list


def main(arg_data_name=None, type="circular"):
    device, device_count = setCuda()
    setSeed()
    data_path, data_name, crop_type, batch_size, num_sample, feature_dim, lr, lr_gamma, num_epochs = parseArgument()
    if arg_data_name is not None:
        data_name = arg_data_name

    train_data_loader, test_data_loader, number_of_class = loadData(type)
    model = setModel(device, feature_dim, number_of_class,
                     model_type="synthesis")

    optimizer, optimizer_init = setOptimizer(device, model, lr)
    lr_scheduler = StepLR(optimizer, step_size=15, gamma=lr_gamma)
    loss_criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):

        model.train()
        average_loss = train(model, train_data_loader, device, loss_criterion,
                             optimizer_init if epoch == 1 else optimizer)
        writer.add_scalar("Loss/average_train_loss", average_loss, epoch)

        correct, total = accuracy(model, device, train_data_loader)
        writer.add_scalar("accuracy/average_train_acc", correct / total, epoch)

        if epoch >= 2:
            lr_scheduler.step()

    #input

    train_color = np.zeros_like(train_data_loader.dataset.y_data, dtype='str')
    train_color[np.where(train_data_loader.dataset.y_data == 0)] = 'b'
    train_color[np.where(train_data_loader.dataset.y_data == 1)] = 'r'

    input_fig = plt.figure(figsize=(10, 10))
    plt.scatter(train_data_loader.dataset.x_data[:, 0], train_data_loader.dataset.x_data[:, 1], c=train_color, s=10)
    plt.scatter(test_data_loader.dataset.x_data[:, 0], test_data_loader.dataset.x_data[:, 1], alpha=0.3, c='#222222', s=10)

    xi = np.linspace(np.min(train_data_loader.dataset.x_data[:, 0]), np.max(train_data_loader.dataset.x_data[:, 0]), 100)
    yi = np.linspace(np.min(train_data_loader.dataset.x_data[:, 1]), np.max(train_data_loader.dataset.x_data[:, 1]), 100)

    X, Y = np.meshgrid(xi, yi)

    zi = model.forward_feature(
        torch.FloatTensor(
            np.concatenate(
                (X.reshape((-1, 1)), Y.reshape((-1, 1))),
                axis=1)
        ).reshape(-1, feature_dim).to(device)
    ).detach().cpu().numpy()
    zi = np.argmax(zi, axis=1)
    zi = zi.reshape(X.shape)

    plt.contourf(X, Y, zi, levels=2, alpha=0.3, cmap="bwr")
    plt.colorbar()
    plt.title("input space")
    #plt.show()


    #embedding
    train_embedding_list, train_label_list = get_embed(model, device, train_data_loader, feature_dim)
    test_embedding_list, _ = get_embed(model, device, test_data_loader, feature_dim)

    train_color = np.zeros_like(train_label_list, dtype='str')
    train_color[np.where(train_label_list == 0)] = 'b'
    train_color[np.where(train_label_list == 1)] = 'r'

    embed_fig = plt.figure(figsize=(10, 10))
    plt.scatter(train_embedding_list[:, 0], train_embedding_list[:, 1], c=train_color, s=10)
    plt.scatter(test_embedding_list[:, 0], test_embedding_list[:, 1], alpha=0.6, c='#222222', s=10)

    xi = np.linspace(np.min(train_embedding_list[:, 0]), np.max(train_embedding_list[:, 0]), 100)
    yi = np.linspace(np.min(train_embedding_list[:, 1]), np.max(train_embedding_list[:, 1]), 100)

    X, Y = np.meshgrid(xi, yi)

    zi = model.forward_feature(
        torch.FloatTensor(
            np.concatenate(
                (X.reshape((-1, 1)), Y.reshape((-1, 1))),
                axis=1)
        ).reshape(-1, feature_dim).to(device)
    ).detach().cpu().numpy()
    zi = np.argmax(zi, axis=1)
    zi = zi.reshape(X.shape)

    plt.contourf(X, Y, zi, levels=2, alpha=0.3, cmap="bwr")
    plt.colorbar()
    plt.title("embedding space")
    #plt.show()



    writer.add_figure('synthesis-input-space', input_fig, num_epochs)
    writer.add_figure('synthesis-embedding-space', embed_fig, num_epochs)



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # "0""None"

if __name__ == '__main__':
    dataset = 'syn'
    print(dataset)
    dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    writer = SummaryWriter("/home/hbyoo/tensorboard/%s/%s" % (dt, dataset))
    main(arg_data_name=dataset, type="circular")
    writer.flush()
    writer.close()
