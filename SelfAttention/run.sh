#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./ja_en_data/kftt-data-1.0/data/orig/kyoto-train.ja --train-tgt=./ja_en_data/kftt-data-1.0/data/orig/kyoto-train.en --dev-src=./ja_en_data/kftt-data-1.0/data/orig/kyoto-dev.ja --dev-tgt=./ja_en_data/kftt-data-1.0/data/orig/kyoto-dev.en --vocab=vocab.json --cuda --lr=1e-4 --patience=1 --valid-niter=2000 --batch-size=16 --dropout=.1
elif [ "$1" = "test" ]; then
	python run.py decode model.bin ./ja_en_data/kftt-data-1.0/data/orig/kyoto-test.ja ./ja_en_data/kftt-data-1.0/data/orig/kyoto-test.en outputs/test_outputs.txt --cuda
elif [ "$1" = "dev" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./ja_en_data/kftt-data-1.0/data/orig/kyoto-dev.ja ./ja_en_data/kftt-data-1.0/data/orig/kyoto-dev.en outputs/dev_outputs.txt --cuda
elif [ "$1" = "train_local" ]; then
	python run.py train --train-src=./ja_en_data/kftt-data-1.0/data/orig/kyoto-train.ja --train-tgt=./ja_en_data/kftt-data-1.0/data/orig/kyoto-train.en --dev-src=./ja_en_data/kftt-data-1.0/data/orig/kyoto-train.ja --dev-tgt=./ja_en_data/kftt-data-1.0/data/orig/kyoto-train.en --vocab=vocab.json --lr=5e-4
elif [ "$1" = "test_local" ]; then
	python run.py decode model.bin ./ja_en_data/split/test.ja ./ja_en_data/split/test.en outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=./ja_en_data/kftt-data-1.0/data/orig/kyoto-test.ja --train-tgt=./ja_en_data/kftt-data-1.0/data/orig/kyoto-test.en vocab.json
elif [ "$1" = "tensorboard" ]; then
	tensorboard --logdir runs --bind_all
else
	echo "Invalid Option Selected"
fi
