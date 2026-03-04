#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=../data/jpn-eng/JESC/train.ja --train-tgt=../data/jpn-eng/JESC/train.en --dev-src=../data/jpn-eng/JESC/dev.ja --dev-tgt=../data/jpn-eng/JESC/dev.en --vocab=vocab.json --cuda --lr=1e-3 --patience=1 --valid-niter=2000 --batch-size=16 --dropout=.2
elif [ "$1" = "test" ]; then
	python run.py decode model.bin ../data/jpn-eng/JESC/test.ja ../data/jpn-eng/JESC/test.en outputs/test_outputs.txt --cuda
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=../data/jpn-eng/JESC/train.ja --train-tgt=../data/jpn-eng/JESC/train.en vocab.json
elif [ "$1" = "tensorboard" ]; then
	tensorboard --logdir runs --bind_all
else
	echo "Invalid Option Selected"
fi
