import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from utils import *

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
MD_DIR = os.path.join(ROOT_DIR, 'data/metadata')

train_info_path = os.path.join(MD_DIR, 'train_info.tsv')
train_rank_path = os.path.join(MD_DIR, 'train_rank.csv')

train_info = pd.read_csv(train_info_path, delimiter='\t')
train_rank = pd.read_csv(train_rank_path)

train_info = train_info.merge(train_rank, on='ID')

artist_occurrence, artist_distinct = get_features_list(train_info['artist_id'].to_list())
composer_occurrence, composer_distinct = get_features_list(train_info['composers_id'].to_list())

train_info['release_time'] = pd.to_datetime(train_info['release_time'])
print(type(train_info['release_time'][0]))
train_info['today'] = pd.to_datetime(datetime.strftime(datetime.today(), "%Y-%m-%d %H:%M:%S"))
train_info['time_interval'] = train_info['today'].sub(train_info['release_time'], axis=0) / np.timedelta64(1, 'D')

train_info['artist_count'] = [get_occurrence(x.artist_id, artist_occurrence)
                              for x in train_info[['artist_id']].itertuples()]
train_info['composers_count'] = [get_occurrence(x.composers_id, composer_occurrence)
                                 for x in train_info[['composers_id']].itertuples()]
