import os
import torch
import torchaudio
from torchaudio import transforms
import librosa
import numpy as np

data, sr = librosa.load('/home/teko/Downloads/test.mp3', mono=True)
chroma_stft = librosa.feature.chroma_stft(data, sr)
print(np.mean(chroma_stft))

