#!/bin/bash

if [ "$1" = "pretrain" ]; then
        CUDA_VISIBLE_DEVICES=0 python pretrain.py pretrain --train-src=../data/jpn-eng/JParaCrawl/train.ja --train-tgt=../data/jpn-eng/JParaCrawl/train.en --dev-src=../data/jpn-eng/JParaCrawl/dev.ja --dev-tgt=../data/jpn-eng/JParaCrawl/dev.en  --save-to=model_pretrain.bin --vocab=vocab.json --cuda --lr=3e-4 --patience=1 --max-num-trial=1 --valid-niter=1000 --batch-size=16 --embed-size=512 --nhead=8 --dropout=0.2
elif [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=../data/jpn-eng/JESC/train.ja --train-tgt=../data/jpn-eng/JESC/train.en --dev-src=../data/jpn-eng/JESC/dev.ja --dev-tgt=../data/jpn-eng/JESC/dev.en --model-path=model_pretrain.bin --save-to=model_finetune.bin --vocab=vocab.json --cuda --lr=3e-4 --patience=1 --valid-niter=2000 --batch-size=32 --embed-size=512 --nhead=8 --dropout=.1 --max-num-trial=5
elif [ "$1" = "test" ]; then
	python run.py decode model_finetune.bin ../data/jpn-eng/JESC/test.ja ../data/jpn-eng/JESC/test.en outputs/test_outputs.txt --cuda
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=../data/jpn-eng/JESC/train.ja --train-tgt=../data/jpn-eng/JESC/train.en vocab.json
elif [ "$1" = "tensorboard" ]; then
	tensorboard --logdir runs --bind_all
else
	echo "Invalid Option Selected"
fi
