#!/bin/bash
FOLDER=./data
if [ ! -d "$FOLDER" ]; then
	mkdir data
fi
cd data
mkdir metadata
cd metadata
wget https://dl.challenge.zalo.ai/hitsong/train_info.tsv
wget https://dl.challenge.zalo.ai/hitsong/train_rank.csv
wget https://dl.challenge.zalo.ai/hitsong/test_info.tsv
