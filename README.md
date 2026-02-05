### Neural Machine Translation Model - Jap2Eng

### Introduction
Simple Chinese-to-English and Japanese-to-English translation models. The codes are **heavily** lifted from Assignment 3 of Stanford's CS224N course on Natural Language Processing with Deep Learning, "Neural Machine Translation". I'm training these Bidirectional 2-layer LSTM for my encoder, and a LSTMCell with Attention for my decoder on my RTX 5070 GPU, with the following architecture and hyperparameters:

### Data
For the Japanese-to-English model, our data comes from the [Japanese-English Subtitle Corpus](https://nlp.stanford.edu/projects/jesc/). The corpus contains 2.8 million sentences, with 2000 dev and 2000 test sentences for evaluation. The data provides data for both English and Japanese sentences separately, so no further processing is required on our end after data download [here](https://www.phontron.com/kftt/#datasystem). 

#### Tokenization
For tokenization, I opted for [SentencePiece from Google](https://github.com/google/sentencepiece) (similar to Assignment 3) which treats each sentence as a sequence of Unicode characters; there is no language-dependent logic. 
This is applied across both the Japanese and English corpus to tokenize the sentences into tokens (via words).

### Model
#### Bidirectional LSTM with Attention
We are using a series of different models to show how model performance varies according to its architecture, using [BLEU](https://en.wikipedia.org/wiki/BLEU). In the first model, we utilised a 2-layer Bidirectional LSTM with Attention, similar to [Luong, Pham and Manning (2015)](https://arxiv.org/abs/1508.04025). I've selected a batch size of 32 for the LSTM, along with a word embedding size of 300 (similar to Word2Vec). In addition, the number of hidden units for the neural layers are set at 512. 

#### Transformer B(asic)
In our benchmark transformer model, we will use Transformers ala [Vaswani et al.](https://arxiv.org/abs/1706.03762). Due to my GPU constraint, I've elected for a smaller batch size (32 instead of the usual 64), and kept the number of encoder and decoder layers constant (at 6). In this model, normalisation is conducted first, prior to residual connection i.e. "Norm and Add" instead of "Add and Norm". Each decoder block contains the following: (Self-attention mechanism, Norm and Add layer, Feed Forward Neural Network with size 2048, and another Norm and Add layer). While our word vector embedding size is relatively small (at 512), we still employ Scaled Dot-Product Attention for normalization.

#### Transformer A(dvanced)
In our second model, we pre-train our Transformer on a separate Japanese to English dataset [Kyoto Free Translation Task by Phontron](https://www.phontron.com/kftt/), while initializing our word vectors using fasttext's pre-trained word vectors. The dataset contains Wikipedia articles (translated from Japanese) related to Kyoto. The data contains ~440k (~300k after cleaning) training sentences, and ~1k dev and test sentences. Pre-training on a dataset drastically different from the JESC should in theory allow our Transformer model to generalize better and faster, while initializing word vectors using fasttest's pre-trained vectors should improve the model's perplexity from the onset. Apart from the embedding size, the layout for this model is similar to Transformer B.

#### Hyperparameters for the Bidirectional LSTM with Attention
1. batch_size: 32
2. word_embeddings: 300 (per word, per language)
3. hidden_units: 256 (for the LSTM layers)
4. dropout_rate: 0.2
5. learning_rate: 5e-4

### Hyperparameters for the Transformer Model B(asic)
1. batch_size: 32
2. word_embeddings: 512
3. dim_feedforward: 2048
4. nhead: 6
5. num_encoder_layer: 6
6. num_decoder_layer: 6
7. dropout_rate: 0.1
8. learning_rate: 3e-4

### Hyperparameters for the Transformer Model A(dvanced)
1. batch_size: 32
2. word_embeddings: 300
3. dim_feedforward: 2048
4. nhead: 6
5. num_encoder_layer: 6
6. num_decoder_layer: 6
7. dropout_rate: 0.1
8. learning_rate: 3e-4

### How To
#### Bidirectional LSTM with Attention
After downloading the data from the JESC website and saaving it to a folder called data in the main directory, run:
1. python preprocess.py

This converts the main data into the training, dev and test datasets.

For the Bidirectional LSTM model, please follow the steps below:

2. python vocab.json --train-src=../data/jpn-eng/JESC/train.ja --train-tgt=../data/jpn-eng/JESC/train.ja vocab.json
This generates the following files: (1) vocab.json (file containing the word2idx and idx2word dictionaries), (2) src.vocab and tgt.vocab files which functions as the lookup table for our Translation model to extract the relevant tokens/ids and (3) src.model and tgt.model, the tokenizer models that splits Japanese and English terms. 

3. sh run.sh train
This trains the Bidirectional LSTM model (from nmt_model.py), using the parameters e.g. size of hidden units, dropout rate, batch size listed in run.sh. Where parameters are not made explicitly available, you may refer to the raw code in nmt_model.py to adjust accordingly.

5. sh run.sh test
Decodes the test input into test output and evaluates the goodness of fit of our test outputs with the actual output using BLEU.

#### Transfomer B(asic)
After downloading the data from the JESC website and saaving it to a folder called data in the main directory, run:
1. python preprocess.py

This converts the main data into the training, dev and test datasets.

For the Transformer B model, please follow the steps below:

2. python vocab.json --train-src=../data/jpn-eng/JESC/train.ja --train-tgt=../data/jpn-eng/JESC/train.ja vocab.json
This generates the following files: (1) vocab.json (file containing the word2idx and idx2word dictionaries), (2) src.vocab and tgt.vocab files which functions as the lookup table for our Translation model to extract the relevant tokens/ids and (3) src.model and tgt.model, the tokenizer models that splits Japanese and English terms. 

3. sh run.sh train
This trains the Transformer model (from nmt_model.py), using the parameters e.g. embedding size, maximum length (since we are using learned positional encoding), dropout rate, batch size listed in run.sh. Where parameters are not made explicitly available, you may refer to the raw code in nmt_model.py to adjust accordingly.

5. sh run.sh test
Decodes the test input into test output and evaluates the goodness of fit of our test outputs with the actual output using BLEU.

### Results
Using the model, I'm getting a [Perplexity score](https://en.wikipedia.org/wiki/Perplexity) of on the dev/holdout dataset, and a BLEU of on the test dataset using a 2-layer Bidirectional LSTM (In Progress!) ala [Luong, Pham and Manning (2015)](https://arxiv.org/abs/1508.04025). As the helper code provided in the Assignment has TensorBoard enabled, we've plotted the performance of the model.


Using our Bidirectional LSTM, we obtained a BLEU of 7.077 on the holdout dataset. On the other hand, our Transformer B(asic) achieved a BLEU of 10.63, a 50% improvement over our LSTM.

### Conclusion
