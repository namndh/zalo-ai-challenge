import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio import transforms
import torchaudio
import glob
from utils import extract_features
import numpy as np


class HitSongDataset(Dataset):
    """ Hit song zalo challeng 2019 dataet"""
    def __init__(self, data_dir, data_info, train=True, transform=None):
        """

        :param data_dir (string): path to directory that store mp3 files
        :param data_info (pandas Dataframe): dataframe that store metadata
        :param train (boolean): status
        :param transform (callable, optional): optional transformation to be applied
            on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data_info = data_info
        self.train = train
        if not os.path.isdir(self.data_dir):
            raise RuntimeError('Dataset not found!')
        self.fns = glob.glob(os.path.join(self.data_dir,'*.mp3'))
        if len(self.fns) == 0:
            raise RuntimeError('No mp3 file was found!')

    def __getitem__(self, index):
        fn = self.fns[index]
        features = extract_features(fn)
        metadata, label = self.get_metadata(fn)
        features = features + np.array(metadata)
        features = torch.from_numpy(features)
        if label == -1:
            return features
        else:
            return features, label

    def __len__(self):
        return len(self.fns)

    def get_metadata(self, fn):
        """

        :param fn (string): path to audio file
        :return: a tensor of metadata information, label of audio file
        """
        sub_pre_idx = 0
        if (self.data_dir[-1] == '/') or (self.data_dir[-1] == '\\'):
            sub_pre_idx = 1
        upper = fn.rfind(self.data_dir)
        lower = fn.find('.mp3')
        audio_id = fn[upper+sub_pre_idx:lower]
        metadata = self.data_info[['time_interval', 'artist_count', 'composers_count', 'label']]\
                    .where(self.data_info['ID'] == audio_id).dropna()
        results = []
        labels = []
        for index, rows in metadata.iterrows():
            result = [rows.time_interval, rows.artist_count, rows.composers_count]
            results.append(result)
            labels.append(rows.label)
        if self.train and len(results) >= 1:
            return results[0], labels[0]
        if self.train and len(results) == 0:
            raise RuntimeError('Can not extract metadata')
        if not self.train and len(results) == 0:
            return [0, 0, 0], -1
        if not self.train and len(results) >= 1:
            return results[0], -1


