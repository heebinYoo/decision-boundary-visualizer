import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np


class SynthesisDataset(Dataset):
    def __init__(self, train=True, data_type="circular"):
        N = 800
        self.train = train
        if train:
            if data_type == "circular":
                theta = np.sqrt(np.random.rand(N)) * 2 * np.pi

                data_a = np.array([np.cos(theta) * 1, np.sin(theta) * 1]).T
                x_a = data_a + np.random.randn(N, 2)

                data_b = np.array([np.cos(theta) * 9, np.sin(theta) * 9]).T
                x_b = data_b + np.random.randn(N, 2)

                self.x_data = np.concatenate((x_a, x_b))
                self.y_data = np.zeros(self.x_data.shape[0])
                self.y_data[:x_a.shape[0]] = 1
                self.y_data[x_a.shape[0]:] = 0
                self.train = train
            elif data_type == "spiral":
                theta = np.sqrt(np.random.rand(N)) * 2 * np.pi

                r_a = 2 * theta + np.pi
                data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
                x_a = data_a + np.random.randn(N, 2)

                r_b = -2 * theta - np.pi
                data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
                x_b = data_b + np.random.randn(N, 2)

                self.x_data = np.concatenate((x_a, x_b))
                self.y_data = np.zeros(self.x_data.shape[0])
                self.y_data[:x_a.shape[0]] = 1
                self.y_data[x_a.shape[0]:] = 0
        else:
            if data_type == "circular":
                theta = np.sqrt(np.random.rand(N)) * 2 * np.pi
                self.x_data = np.array([np.cos(theta) * 4, np.sin(theta) * 4]).T + np.random.randn(N, 2)
            elif data_type == "spiral":
                theta = np.sqrt(np.random.rand(N)) * 2 * np.pi
                r_a = 1 * theta + np.pi
                data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
                x_a = data_a + np.random.randn(N, 2)

                r_b = -1 * theta - np.pi
                data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
                x_b = data_b + np.random.randn(N, 2)

                self.x_data = np.concatenate((x_a, x_b))

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.x_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.LongTensor([self.y_data[idx]]) if self.train else -1
        return x, y


