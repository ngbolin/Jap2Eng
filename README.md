### Neural Machine Translation Model - Jap2Eng

### Introduction
Simple Chinese-to-English and Japanese-to-English translation models. The codes are **heavily** lifted from Assignment 3 of Stanford's CS224N course on Natural Language Processing with Deep Learning, "Neural Machine Translation". I'm training these Bidirectional 2-layer LSTM for my encoder, and a LSTMCell with Attention for my decoder on my RTX 5070 GPU, with the following architecture and hyperparameters:

### Data
For the Japanese-to-English model, our data comes from the [Kyoto Free Translation Task (KFTT)](https://www.phontron.com/kftt/). The corpus contains ~440k sentences, with ~1.1k dev and test sentences for evaluation. The data provides data for both English and Japanese sentences separately, so no further processing is required on our end after data download [here](https://www.phontron.com/kftt/#datasystem). 

#### Tokenization
For tokenization, I opted for [SentencePiece from Google](https://github.com/google/sentencepiece) (similar to Assignment 3) which treats each sentence as a sequence of Unicode characters; there is no language-dependent logic. 
This is applied across both the Japanese and English corpus to tokenize the sentences into tokens (via words).

### Model
We are using a series of different models to show how model performance varies according to its architecture, using [BLEU](https://en.wikipedia.org/wiki/BLEU). In the first model, we utilised a 2-layer Bidirectional LSTM with Attention, similar to [Luong, Pham and Manning (2015)](https://arxiv.org/abs/1508.04025). In our second model, we will use Transformers ala [Vaswani et al.](https://arxiv.org/abs/1706.03762).

#### Hyperparameters
1. batch_size: 32
2. word_embeddings: 300 (per word, per language)
3. hidden_units: 256 (for the LSTM layers)
5. dropout_rate: 0.5

### How To
After forking the repository, use the following to train and test the model.

### Results
Using the model, I'm getting a [Perplexity score](https://en.wikipedia.org/wiki/Perplexity) of ~8 on the Dev dataset, and a BLEU of 15.4 on the Test dataset. Going forward, I intend to implement a 2-layer Bidirectional LSTM (In Progress!) ala [Luong, Pham and Manning (2015)](https://arxiv.org/abs/1508.04025) and evaluate its performance vis-a-vis a single-layer model. Finally, I will implement a Neural Machine Translation model using Transformers.

### Conclusion
