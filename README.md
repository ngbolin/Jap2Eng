### Neural Machine Translation Model - Jap2Eng

### Introduction
Simple Chinese-to-English and Japanese-to-English translation models. The codes are **heavily** lifted from Assignment 3 of Stanford's CS224N course on Natural Language Processing with Deep Learning, "Neural Machine Translation". I'm training these Bidirectional 2-layer LSTM for my encoder, and a LSTMCell with Attention for my decoder on my RTX 5070 GPU, with the following architecture and hyperparameters:

### Data
For the Japanese-to-English model, our data comes from the [Kyoto Free Translation Task (KFTT)](https://www.phontron.com/kftt/). The corpus contains ~440k sentences, with ~1.1k dev and test sentences for evaluation. The data provides data for both English and Japanese sentences separately, so no further processing is required on our end after data download [here](https://www.phontron.com/kftt/#datasystem). 

#### Tokenization
For tokenization, I opted for [SentencePiece from Google](https://github.com/google/sentencepiece) (similar to Assignment 3) which treats each sentence as a sequence of Unicode characters; there is no language-dependent logic. 
This is applied across both the Japanese and English corpus to tokenize the sentences into tokens (via words).

### Model
#### Bidirectional LSTM
We are using a series of different models to show how model performance varies according to its architecture, using [BLEU](https://en.wikipedia.org/wiki/BLEU). In the first model, we utilised a 2-layer Bidirectional LSTM with Attention, similar to [Luong, Pham and Manning (2015)](https://arxiv.org/abs/1508.04025). I've selected a batch size of 32 for the LSTM, along with a word embedding size of 300 (similar to Word2Vec). In addition, the number of hidden units for the neural layers are set at 256. 

#### Transformer A
In our second model, we will use Transformers ala [Vaswani et al.](https://arxiv.org/abs/1706.03762). Due to my GPU constraint, I've elected for a smaller feedforward size (512 instead of 2048), but kept the number of encoder and decoder layers constant (at 6). In this model, normalisation is conducted first, prior to residual connection i.e. "Norm and Add" instead of "Add and Norm". Each decoder block contains the following: (Self-attention mechanism, Norm and Add layer, Feed Forward Neural Network with size 512, and another Norm and Add layer). While our word vector embedding size is pretty low (at 300), we still employ Scaled Dot-Product Attention for normalization.

#### Hyperparameters for the Bidirectional LSTM
1. batch_size: 32
2. word_embeddings: 300 (per word, per language)
3. hidden_units: 256 (for the LSTM layers)
4. dropout_rate: 0.2
5. learning_rate: 5e-4

### Hyperparameters for the Transformer Model A
1. batch_size: 16
2. word_embeddings: 300
3. dim_feedforward: 512
4. nhead: 6
5. num_encoder_layer: 6
6. num_decoder_layer: 6
7. dropout_rate: 0.1
8. learning_rate: 3e-4

### Hyperparameters for the Transformer Model B
1. batch_size: 8
2. word_embeddings: 300
3. dim_feedforward: 1024
4. nhead: 6
5. num_encoder_layer: 6
6. num_decoder_layer: 6
7. dropout_rate: 0.1
8. learning_rate: 3e-4

### How To
For each model: BidirectionalLSTM and SelfAttention, follow the 

### Results
Using the model, I'm getting a [Perplexity score](https://en.wikipedia.org/wiki/Perplexity) of 8 on the dev/holdout dataset, and a BLEU of 15.4 on the test dataset using a 2-layer Bidirectional LSTM (In Progress!) ala [Luong, Pham and Manning (2015)](https://arxiv.org/abs/1508.04025). As the helper code provided in the Assignment has TensorBoard enabled, we've plotted the performance of the model.

![TensorBoard for LSTM](https://github.com/ngbolin/Jap2Eng/blob/main/SelfAttention/images/TensorBoard%20Process%20Monitoring.png)


For the Transformer A, we obtained a Perplexity of 11 on the dev/holdout dataset, and a BLEU of XX on the test dataset.

### Conclusion
