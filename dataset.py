import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio import transforms
import torchaudio
import glob
from utils import extract_features
import numpy as np
from datetime import datetime

class HitSongDataset(Dataset):
    """ Hit song zalo challeng 2019 dataet"""
    def __init__(self, data_dir, train_data_info, train=True, test_data_info=None, transform=None):
        """

        :param data_dir (string): path to directory that store mp3 files
        :param data_info (pandas Dataframe): dataframe that store metadata
        :param train (boolean): status
        :param transform (callable, optional): optional transformation to be applied
            on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.train_data_info = train_data_info
        self.test_data_info = test_data_info
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
        audio_id = int(audio_id)
        if self.train:
            metadata = self.train_data_info[['time_interval', 'artist_count', 'composers_count', 'label']] \
                .where(self.train_data_info['ID'] == audio_id).dropna()
            results = []
            labels = []
            for index, rows in metadata.iterrows():
                result = [rows.time_interval, rows.artist_count, rows.composers_count]
                results.append(result)
                labels.append(rows.label)
            if len(results) >= 1:
                return results[0], labels[0]
            if len(results) == 0:
                raise RuntimeError('Can not extract metadata')
        if not self.train:
            artist_count = 0
            composers_count = 0

            artist_id = self.test_data_info['artist_id']\
                .where(self.test_data_info['ID'] == audio_id).dropna().iloc[0]
            composers_id = self.test_data_info['composers_id']\
                .where(self.test_data_info['ID'] == audio_id).dropna().iloc[0]
            time_release = self.test_data_info['release_date']\
                .where(self.test_data_info['ID'] == audio_id).dropna().iloc[0]
            time_release = datetime.strptime(time_release, '%Y-%m-%d %H:%M:%S')
            time_interval = datetime.today() - time_release
            time_interval = time_interval.days
            if artist_id in self.train_data_info['artist_id'].tolist():
                artist_count = self.train_data_info['artist_count']\
                    .where(self.train_data_info['artist_id'] == artist_id).dropna().iloc[0]
                artist_count = float(artist_count)
            if composers_id in self.train_data_info['composers_id'].tolist():
                composers_count = self.train_data_info['composers_count']\
                    .where(self.train_data_info['composers_id'] == composers_id).dropna().iloc[0]
            return [time_interval, artist_count, composers_count]

