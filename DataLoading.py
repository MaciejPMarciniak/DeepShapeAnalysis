import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


# --- DL Dataset ---
class PointCloudDataset(Dataset):

    def __init__(self, point_cloud_dir, info_file, transform=None):
        self.df_mesh_info = pd.read_csv(os.path.join(point_cloud_dir, info_file), header=0)
        self.point_cloud_dir = point_cloud_dir
        self.transform = transform

    def __len__(self):
        return len(self.df_mesh_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        point_cloud = self.get_point_cloud(idx)
        description = self.df_mesh_info.iloc[idx].to_dict()
        sample = {'point_cloud': point_cloud, 'description': description}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_point_cloud(self, index):
        mesh_name = os.path.join(self.point_cloud_dir, 'PointClouds', 'point_cloud_' + str(index).zfill(5) + '.csv')
        point_cloud = np.loadtxt(mesh_name, delimiter=',')
        return point_cloud


class Ravel(object):

    def __call__(self, sample):
        sample['point_cloud'] = np.ravel(sample['point_cloud'])
        return sample


class ToTensor(object):

    def __call__(self, sample):
        point_cloud, description = sample['point_cloud'], sample['description']
        return {'point_cloud': torch.from_numpy(point_cloud).float(), 'description': description}


class Transpose(object):

    def __call__(self, sample):
        sample['point_cloud'] = sample['point_cloud'].T
        return sample


class DataLoading:

    def __init__(self, data_path, mesh_info_file, batch_size, validation_split, ravel=True, seed=42):
        self.data_path = data_path
        self.mesh_info_file = mesh_info_file
        self.point_cloud_dataset = self.create_point_cloud_dataset() if ravel \
            else self.create_unravelled_point_cloud_dataset()
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.seed = seed

    def create_point_cloud_dataset(self):
        point_cloud_dataset = PointCloudDataset(self.data_path, self.mesh_info_file,
                                                transform=transforms.Compose([
                                                    Ravel(),
                                                    ToTensor()]))
        return point_cloud_dataset

    def create_unravelled_point_cloud_dataset(self):
        point_cloud_dataset = PointCloudDataset(self.data_path, self.mesh_info_file,
                                                transform=transforms.Compose([
                                                    Transpose(),
                                                    ToTensor()]))
        return point_cloud_dataset

    def get_shuffled_indices(self, dataset_size):
        indices = list(range(dataset_size))
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        return indices

    def split_indices(self):
        dataset_size = len(self.point_cloud_dataset)
        split = int(np.floor(self.validation_split * dataset_size))
        indices = self.get_shuffled_indices(dataset_size)
        return indices[split:], indices[:split]

    def build_shuffled_training_and_validation_data(self):
        train_indices, val_indices = self.split_indices()

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(self.point_cloud_dataset, batch_size=self.batch_size,
                                  sampler=train_sampler, num_workers=0, pin_memory=True)
        validation_loader = DataLoader(self.point_cloud_dataset, batch_size=self.batch_size,
                                       sampler=valid_sampler, num_workers=0, pin_memory=True)
        return train_loader, validation_loader

    def build_whole_dataset_loader(self):
        dataset_loader = DataLoader(self.point_cloud_dataset, batch_size=self.batch_size, num_workers=0)
        return dataset_loader
