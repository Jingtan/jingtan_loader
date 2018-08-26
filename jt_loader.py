#!G:\Users\jingtan\Anaconda3\python.exe
# -*- coding: utf-8 -*-

import os
import os.path
import torch.utils.data as data
import torch


class JtLoader(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            self.train_data, self.train_labels = torch.load(self.root)
        else:
            self.test_data, self.test_labels = torch.load(self.root)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
#         img = Image.fromarray(img.numpy(), mode='L')

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


print('ok!')
