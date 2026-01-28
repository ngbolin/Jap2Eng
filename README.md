### Neural Machine Translation Model - Jap2Eng

### Introduction
Simple Chinese-to-English and Japanese-to-English translation models. The codes are **heavily** lifted from Assignment 3 of Stanford's CS224N course on Natural Language Processing with Deep Learning, "Neural Machine Translation". I'm training these Bidirectional LSTMs with Attention on my RTX 5070 GPU, with the following architecture and hyperparameters:

#### Architecture

#### Hyperparameters
1. batch_size: 32
2. word_embeddings: 300 (per word, per language)
3. hidden_units: 256 (for the LSTM layers)
5. dropout_rate: 0.5

### Data
For the Japanese-to-English model, our data comes from the [Japanese English Subtitle Corpus (JESC)](https://nlp.stanford.edu/projects/jesc/]). The corpus contains 2.8 million sentences, with 2000 dev and test sentences for evaluation.

#### Tokenization
For tokenization, I opted for [SentencePiece from Google](https://github.com/google/sentencepiece) (similar to Assignment 3) which treats each sentence as a sequence of Unicode characters; there is no language-dependent logic. 
This is applied across both the Japanese and English corpus to tokenize the sentences into tokens (via words).

### Model

### How To

### Results

### Conclusion
