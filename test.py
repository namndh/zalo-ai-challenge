import os
import torch
import torchaudio
from torchaudio import transforms

data, sample_rate = torchaudio.load('/home/teko/Downloads/test.mp3')
print(data.shape)
print(sample_rate)

amp_db = transforms.AmplitudeToDB(stype='magnitude')
db = amp_db.forward(data)
print(db.shape)
https://gist.github.com/parulnith/7f8c174e6ac099e86f0495d3d9a4c01e#file-untitled9-ipynb