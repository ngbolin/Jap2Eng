### Neural Machine Translation Model - Jap2Eng

### Introduction
Simple Chinese-to-English and Japanese-to-English translation models. The codes are **heavily** lifted from Assignment 3 of Stanford's CS224N course on Natural Language Processing with Deep Learning, "Neural Machine Translation". I'm training these Bidirectional 2-layer LSTM for my encoder, and a LSTMCell with Attention for my decoder on my RTX 5070 GPU, with the following architecture and hyperparameters:

### Data
For the Japanese-to-English model, our data comes from the [Japanese-English Subtitle Corpus](https://nlp.stanford.edu/projects/jesc/). The corpus contains 2.8 million sentences, with 2000 dev and 2000 test sentences for evaluation. The data provides data for both English and Japanese sentences separately, so no further processing is required on our end after data download [here](https://www.phontron.com/kftt/#datasystem). 

#### Tokenization
For tokenization, I opted for [SentencePiece from Google](https://github.com/google/sentencepiece) (similar to Assignment 3) which treats each sentence as a sequence of Unicode characters; there is no language-dependent logic. 
This is applied across both the Japanese and English corpus to tokenize the sentences into tokens (via words).

### Evaluation Metric
To evaluate our model, we will use [BLEURT](https://github.com/google-research/bleurt), a Transfer Learning-Based Metric for Natural Language Generation. BLEURT takes a sentence pair (reference and candidate) and returns a score that indicates the extent to which the candidate is fluent and conveys the meaning of the reference. We will use the recommended checkpoint of BLEURT-20, which returns a score between 0 and 1 where 0 indicates a random output and 1 a perfect one. Following Google Research's recommendation, we will average the BLEURT scores across the sentences in the corpus.

--------------------------------------------------------------------------------------------------------------------------------

### Model
--------------------------------------------------------------------------------------------------------------------------------
#### Bidirectional LSTM with Attention
We are using a series of different models to show how model performance varies according to its architecture, using [BLEU](https://en.wikipedia.org/wiki/BLEU). In the first model, we utilised a 2-layer Bidirectional LSTM with Attention, similar to [Luong, Pham and Manning (2015)](https://arxiv.org/abs/1508.04025). I've selected a batch size of 32 for the LSTM, along with a word embedding size of 300 (similar to Word2Vec). In addition, the number of hidden units for the neural layers are set at 512. 

#### Hyperparameters for the Bidirectional LSTM with Attention
1. batch_size: 32
2. word_embeddings: 300 (per word, per language)
3. hidden_units: 256 (for the LSTM layers)
4. dropout_rate: 0.2
5. learning_rate: 5e-4
6. num_trials: 5

--------------------------------------------------------------------------------------------------------------------------------

#### Transformer B(asic)
In our benchmark transformer model, we will use Transformers ala [Vaswani et al.](https://arxiv.org/abs/1706.03762). Due to my GPU constraint, I've elected for a smaller batch size (32 instead of the usual 64), and kept the number of encoder and decoder layers constant (at 6). In this model, normalisation is conducted first, prior to residual connection i.e. "Norm and Add" instead of "Add and Norm". Each decoder block contains the following: (Self-attention mechanism, Norm and Add layer, Feed Forward Neural Network with size 2048, and another Norm and Add layer). While our word vector embedding size is relatively small (at 512), we still employ Scaled Dot-Product Attention for normalization.

#### Hyperparameters for the Transformer Model B(asic)
1. batch_size: 32
2. word_embeddings: 512
3. dim_feedforward: 2048
4. nhead: 8
5. num_encoder_layer: 6
6. num_decoder_layer: 6
7. dropout_rate: 0.1
8. learning_rate: 3e-4
9. num_trials: 5

--------------------------------------------------------------------------------------------------------------------------------

#### Transformer A(dvanced)
In our second model, we pre-train our Transformer on a separate Japanese to English dataset For pre-training, we will use data from [JParaCrawl](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/), the largest publicly available English-Japanese parallel corpus created by NTT. We create a function, ```preprocess.py``` to extract the first 2 million sentence-pairs, and allocate 99.9% of them for training (and 0.1% for dev). Pre-training on a dataset that is different from the JESC should in theory allow our Transformer model to generalize better and faster, and setting the parameter ```num_trials``` to 1 prevents the Pretrained Transformer from being overly fixated on the Pretraining dataset. Apart from the use of the Pretrained Transformer, the hyperparameters for TransformerA and TransformerB are largely similar.

#### Hyperparameters for Pretrained Transformer
1. batch_size: 16
2. word_embeddings: 512
3. dim_feedforward: 2048
4. nhead: 6
5. num_encoder_layer: 6
6. num_decoder_layer: 6
7. dropout_rate: 0.1
8. learning_rate: 3e-4
9. num_trials: 1

#### Hyperparameters for the Transformer Model A(dvanced)
1. batch_size: 32
2. word_embeddings: 512
3. dim_feedforward: 2048
4. nhead: 6
5. num_encoder_layer: 6
6. num_decoder_layer: 6
7. dropout_rate: 0.1
8. learning_rate: 3e-4
9. num_trials: 5

--------------------------------------------------------------------------------------------------------------------------------

### How To

#### Bidirectional LSTM and Transformer B(asic)  
The steps for all models are (largely similar) apart from the arguments listed. In the Bidirectional LSTM with Attention, the argument ```hidden_units``` is made available for the number of hidden units for each word. On the other hand, the parameter ```nheads``` is made available for the Transformer models due to the usage of Multihead Attention. 

This generates the following files: (1) vocab.json (file containing the word2idx and idx2word dictionaries), (2) src.vocab and tgt.vocab files which functions as the lookup table for our Translation model to extract the relevant tokens/ids and (3) src.model and tgt.model, the tokenizer models that splits Japanese and English terms. 

1. ```python vocab.json --train-src=../data/jpn-eng/JESC/train.ja --train-tgt=../data/jpn-eng/JESC/train.ja vocab.json```

This trains the models (in [nmt_model.py]), using the parameters e.g. embedding size, nheads, dropout rate, batch size listed in [run.sh]. Where parameters are not made explicitly available, you may refer to the raw code in [nmt_model.py] to adjust accordingly.

2. ```sh run.sh train```

Decodes the test input into test output and evaluates the goodness of fit of our test outputs with the actual output using BLEU.

3. ```sh run.sh test```

--------------------------------------------------------------------------------------------------------------------------------

#### Bidirectional LSTM and Transformer A(dvanced)

For pre-training, we will use data from [JParaCrawl](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/), the largest publicly available English-Japanese parallel corpus created by NTT. The specific version we are using is V2.0, which contains 10.0 million sentence pairs. We have created a function, called ```preprocess.py``` that processes and creates 2 different datasets for each language (training and dev).
1. ```python preprocess.py```

The following function generates the following files: (1) vocab.json (file containing the word2idx and idx2word dictionaries), (2) src.vocab and tgt.vocab files which functions as the lookup table for our Translation model to extract the relevant tokens/ids and (3) src.model and tgt.model, the tokenizer models that splits Japanese and English terms. 

2. ```python vocab.json --train-src=../data/jpn-eng/JESC/train.ja --train-tgt=../data/jpn-eng/JESC/train.ja vocab.json```

After data pre-processing, we proceed to pretrain the model. This cal be done by calling the following function:
3. ```sh run.sh pretrain```

This trains the models (in [nmt_model.py]), using the parameters e.g. embedding size, nheads, dropout rate, batch size listed in [run.sh]. Where parameters are not made explicitly available, you may refer to the raw code in [nmt_model.py] to adjust accordingly.

4. ```sh run.sh train```

Decodes the test input into test output and evaluates the goodness of fit of our test outputs with the actual output using BLEU.

6. ```sh run.sh test```



--------------------------------------------------------------------------------------------------------------------------------

### Results
Using the model, I'm getting a BLEURT of XX on the test dataset using a 2-layer Bidirectional LSTM ala [Luong, Pham and Manning (2015)](https://arxiv.org/abs/1508.04025). As the helper code provided in the Assignment has TensorBoard enabled, we've plotted the performance of the model.


Using our Bidirectional LSTM, we obtained a BLEURT of XX on the holdout dataset. Our Transformer B(asic) achieved a BLEURT of 0.1970. On the other hand, our Transformer A(dvanced) achieved a BLEURT of 0.4800.

### Conclusion
