import os
import sys
import numpy as np
import pandas as pd
import librosa
import torch

def get_features_list(id_list):
    result = []
    for i in id_list:
        if ',' in i:
            result.extend(i.split(','))
        elif '.' in i:
            result.extend(i.split('.'))
        else:
            result.append(i)
    return result, list(set(result))


def get_occurrence(data, data_list):
    result = 0
    i = 0
    if '.' in data:
        data = data.split('.')
        for e in data:
            i += 1
            result += data_list.count(e)
        result = result/i
    elif ',' in data:
        data = data.split(',')
        for e in data:
            i += 1
            result += data_list.count(e)
        result = result/i
    else:
        result = data_list.count(data)

    return result


def extract_features(fn):
    y, sr = librosa.load(fn, mono=True)
    chroma_stft = librosa.feature.chroma_stft(y,sr)
    spec_cent = librosa.feature.spectral_centroid(y, sr)
    spec_bw = librosa.feature.spectral_bandwidth(y, sr)
    rolloff = librosa.feature.spectral_rolloff(y, sr)
    zcr = librosa.feature.zero_crossing_rate(y, sr)
    mfcc = librosa.feature.mfcc(y, sr)

    return [np.mean(chroma_stft), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff),
            np.mean(zcr), np.mean(mfcc)]


class RMSELoss(torch.nn.Module):
    """
        Root Mean Squared Error for pytorch network
    """
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, predicts, labels):
        return torch.sqrt(self.mse(predicts, y))



