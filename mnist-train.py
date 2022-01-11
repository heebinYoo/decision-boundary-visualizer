# coding=utf-8
import os

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
    parser.add_argument('--feature_dim', default=3, type=int, help='feature dim')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='learning rate scheduler gamma')
    parser.add_argument('--num_epochs', default=3, type=int, help='train epoch number')

    opt = parser.parse_args()

    return opt.data_path, opt.data_name, opt.crop_type, opt.batch_size, opt.num_sample, opt.feature_dim, opt.lr, opt.lr_gamma, opt.num_epochs


def loadData(batch_size, num_sample, seen=(0, 1), unseen=(4, 5)):
    mnist_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))
    ])
    train_data_set = MNIST(root='/data/hbyoo',  # 다운로드 경로 지정
                           train=True,  # True를 지정하면 훈련 데이터로 다운로드
                           transform=mnist_transform,  # 텐서로 변환
                           download=True)

    seen_idx = torch.cat((
        torch.nonzero(train_data_set.targets == seen[0]).reshape(-1),
        torch.nonzero(train_data_set.targets == seen[1]).reshape(-1),
        torch.nonzero(train_data_set.targets == seen[2]).reshape(-1),
        torch.nonzero(train_data_set.targets == seen[3]).reshape(-1)
    )).reshape(-1)

    unseen_idx = torch.cat((
        torch.nonzero(train_data_set.targets == unseen[0]).reshape(-1),
        torch.nonzero(train_data_set.targets == unseen[1]).reshape(-1)
    )).reshape(-1)

    seen_data = Subset(train_data_set, seen_idx)
    seen_label = train_data_set.targets[seen_idx]
    unseen_data = Subset(train_data_set, unseen_idx)

    train_data_loader = DataLoader(seen_data, batch_size, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(unseen_data, batch_size, shuffle=False,
                                  num_workers=4)

    return train_data_loader, test_data_loader, 10


def setModel(device, feature_dim, number_of_class, model_type):
    model = ConfidenceControl(feature_dim, number_of_class, model_type)
    # writer.add_graph(model)
    return model.to(device)


def setOptimizer(model, lr):
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
            correct += torch.eq(pred, labels).sum().float().item()

    return correct, total


def train(net, train_data_loader, device, criterion, optimizer):
    total_loss = 0
    iternum = 0
    for inputs, labels in tqdm(train_data_loader, dynamic_ncols=True):
        inputs, labels = inputs.to(device), labels.to(device)

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


def main(arg_data_name=None):
    device, device_count = setCuda()
    setSeed()
    data_path, data_name, crop_type, batch_size, num_sample, feature_dim, lr, lr_gamma, num_epochs = parseArgument()
    if arg_data_name is not None:
        data_name = arg_data_name

    seen = (0, 1, 2, 3)
    unseen = (4, 5)
    train_data_loader, test_data_loader, number_of_class = loadData(batch_size, num_sample, seen, unseen)
    model = setModel(device, feature_dim, number_of_class,
                     model_type="mnist")

    optimizer, optimizer_init = setOptimizer(model, lr)
    lr_scheduler = StepLR(optimizer, step_size=15, gamma=lr_gamma)
    loss_criterion = nn.CrossEntropyLoss()

    model.eval()
    train_embedding_list, train_label_list = get_embed(model, device, train_data_loader, feature_dim)
    test_embedding_list, test_label_list = get_embed(model, device, test_data_loader, feature_dim)
    s0 = torch.from_numpy(
        train_embedding_list[np.where(train_label_list == seen[0]), :].reshape(-1, 3).astype("float")).unsqueeze(0)
    s1 = torch.from_numpy(
        train_embedding_list[np.where(train_label_list == seen[1]), :].reshape(-1, 3).astype("float")).unsqueeze(0)
    s2 = torch.from_numpy(
        train_embedding_list[np.where(train_label_list == seen[2]), :].reshape(-1, 3).astype("float")).unsqueeze(0)
    s3 = torch.from_numpy(
        train_embedding_list[np.where(train_label_list == seen[3]), :].reshape(-1, 3).astype("float")).unsqueeze(0)
    u0 = torch.from_numpy(
        test_embedding_list[np.where(test_label_list == unseen[0]), :].reshape(-1, 3).astype("float")).unsqueeze(0)
    u1 = torch.from_numpy(
        test_embedding_list[np.where(test_label_list == unseen[1]), :].reshape(-1, 3).astype("float")).unsqueeze(0)

    label = np.concatenate(
        (np.array(s0.shape[1] * [0]).reshape(-1), np.array(s1.shape[1] * [1]).reshape(-1),
         np.array(s2.shape[1] * [2]).reshape(-1), np.array(s3.shape[1] * [3]).reshape(-1),
         np.array(u0.shape[1] * [4]).reshape(-1), np.array(u1.shape[1] * [5]).reshape(-1)))
    label = torch.as_tensor(label).unsqueeze(0)

    writer.add_embedding(torch.cat((s0, s1, s2, u0, u1), dim=1).resg,
                         metadata=label)





    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.scatter(train_embedding_list[np.where(train_label_list == seen[0]), 0],
    #            train_embedding_list[np.where(train_label_list == seen[0]), 1],
    #            train_embedding_list[np.where(train_label_list == seen[0]), 2],
    #            c='r', alpha=0.1, s=100,
    #            label='seen1')
    # ax.scatter(train_embedding_list[np.where(train_label_list == seen[1]), 0],
    #            train_embedding_list[np.where(train_label_list == seen[1]), 1],
    #            train_embedding_list[np.where(train_label_list == seen[1]), 2],
    #            c='b', alpha=0.1, s=100,
    #            label='seen2')
    # ax.scatter(train_embedding_list[np.where(train_label_list == seen[2]), 0],
    #            train_embedding_list[np.where(train_label_list == seen[2]), 1],
    #            train_embedding_list[np.where(train_label_list == seen[2]), 2],
    #            c='#41f1f1', alpha=0.1, s=100,
    #            label='seen3')
    # ax.scatter(test_embedding_list[np.where(test_label_list == unseen[0]), 0],
    #            test_embedding_list[np.where(test_label_list == unseen[0]), 1],
    #            test_embedding_list[np.where(test_label_list == unseen[0]), 2],
    #            c='g', alpha=0.5, s=5,
    #            label='unseen1')
    # ax.scatter(test_embedding_list[np.where(test_label_list == unseen[1]), 0],
    #            test_embedding_list[np.where(test_label_list == unseen[1]), 1],
    #            test_embedding_list[np.where(test_label_list == unseen[1]), 2],
    #            c='y', alpha=0.5, s=5,
    #            label='unseen2')
    # plt.legend()
    #
    # writer.add_figure('mnist-embedding-space', fig,  0)

    for epoch in range(num_epochs):

        model.train()
        average_loss = train(model, train_data_loader, device, loss_criterion,
                             optimizer_init if epoch == 1 else optimizer)
        writer.add_scalar("Loss/average_train_loss", average_loss, epoch)

        if epoch >= 2:
            lr_scheduler.step()




os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # "0""None"

if __name__ == '__main__':
    # for dataset in ['car', 'cub', 'sop', 'isc', 'mnist']:
    dataset = 'mnist'
    print(dataset)
    dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    writer = SummaryWriter("/home/hbyoo/tensorboard/%s/%s" % (dt, dataset))
    main(arg_data_name=dataset)
    writer.flush()
    writer.close()
