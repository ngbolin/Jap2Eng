import torch
import math
import torch.nn as nn

class ModelEmbeddings(nn.Module): 
    def __init__(self, embed_size, vocab, max_len=512):
        """
        Init the Embedding layers.

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size

        # default values
        src_pad_token_idx = vocab.src['<pad>']
        tgt_pad_token_idx = vocab.tgt['<pad>']

        self.source_layer = nn.Embedding(len(vocab.src), embed_size, src_pad_token_idx)
        self.target_layer = nn.Embedding(len(vocab.tgt), embed_size, tgt_pad_token_idx)
        
        self.pos_embeddings = nn.Embedding(max_len, embed_size)

    def forward(self, input_tensor, is_source=True):
        """
        @param input_tensor: Tensor of token IDs (batch_size, seq_len)
        @param is_source: Boolean, if True use source embeddings, else target
        """
        # 1. Choose the correct embedding layer
        embed_layer = self.source_layer if is_source else self.target_layer
        x = embed_layer(input_tensor) # (batch_size, seq_len, embed_size)
        
        # Normalisation
        x = x * math.sqrt(self.embed_size)
        
        batch_size, seq_len = input_tensor.size()
        assert seq_len <= self.pos_embeddings.num_embeddings
        
        # 2. Add learned positional embeddings
        positions = torch.arange(seq_len, device=input_tensor.device).unsqueeze(0)
        
        pos_emb = self.pos_embeddings(positions)  # (1, seq_len, embed_size)
        return x + pos_emb