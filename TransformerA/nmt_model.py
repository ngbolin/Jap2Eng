from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import math

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """

    def __init__(self, vocab, embed_size=512, max_len=1024, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dropout_rate=0.1):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size, the size of hidden states (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        self.dropout_rate=dropout_rate
        self.vocab=vocab
        self.nhead=nhead
        self.num_encoder_layers=num_encoder_layers
        self.num_decoder_layers=num_decoder_layers
        self.dim_feedforward=4*embed_size
        self.embed_size = embed_size

        # default values
        src_pad_token_idx = self.vocab.src['<pad>']
        tgt_pad_token_idx = self.vocab.tgt['<pad>']

        self.source_layer = nn.Embedding(len(self.vocab.src), embed_size, src_pad_token_idx)
        self.target_layer = nn.Embedding(len(self.vocab.tgt), embed_size, tgt_pad_token_idx)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.pos_embeddings = nn.Embedding(max_len, embed_size) # (S, E)

        # We are using nhead=6 because our embedding size is 300
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dropout=self.dropout_rate,
            dim_feedforward=self.dim_feedforward,
            norm_first=True
        )

        self.target_vocab_projection=nn.Linear(embed_size, len(self.vocab.tgt))

        self.target_vocab_projection.weight = self.target_layer.weight

    def encode(self, source_padded):
        """ Encodes a source sequence (padded) using the Encoder from the Transformer model.

        @param source_padded (Tensor): a tensor of shape (s, b) where s refers to the source_length and b refers to the batch size
        """
        # source_padded: (S, B)
        seq_len = source_padded.size(0)
        source_embeddings = self.source_layer(source_padded) # (S, B, E)
        source_embeddings = source_embeddings * math.sqrt(self.embed_size)
    
        # Use (S, 1) indices for positional embeddings
        pos_indices = torch.arange(seq_len, device=self.device).unsqueeze(1) 
        source_embeddings = source_embeddings + self.pos_embeddings(pos_indices) # (S, B, E)

        return self.transformer.encoder(source_embeddings) # (S, B, E)

    def decode(self, target_padded, memory):
        """ Decodes the target sequence (padded) using the Decoder from the Transformer model. Employs a target mask to ignore positions > t.

        @param target_padded (Tensor): a tensor of shape (t, b) where s refers to the target_length and b refers to the batch size
        """
        # target_padded: (T, B)
        seq_len = target_padded.size(0)
        target_embeddings = self.target_layer(target_padded) # (T, B, E)
        target_embeddings = target_embeddings * math.sqrt(self.embed_size)

         # Use (T, 1) indices for positional embeddings
        pos_indices = torch.arange(seq_len, device=self.device).unsqueeze(1) 
        target_embeddings = target_embeddings + self.pos_embeddings(pos_indices) # (T, B, E)
    
        # Causal mask for the target sequence (T, T)
        target_mask = self.transformer.generate_square_subsequent_mask(seq_len).to(self.device) 
    
        return self.transformer.decoder(target_embeddings, memory, tgt_mask=target_mask) # (T, B, E)
        

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists of source and target sentences into padded tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)  # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)  # Tensor: (tgt_len, b)
        src_len, batch_size = source_padded.size() 
        tgt_len, batch_size = target_padded.size()

        assert target_padded.max().item() < len(self.vocab.tgt)
        assert target_padded.min().item() >= 0

        # Employ word embeddings for both source and target sequences
        source_embeddings = self.source_layer(source_padded) # (src_len, b, embed_size)
        target_embeddings = self.target_layer(target_padded) # (tgt_len, b, embed_size)
        
        # Conduct normalisation
        source_embeddings = source_embeddings * math.sqrt(self.embed_size) # (src_len, b, embed_size)
        target_embeddings = target_embeddings * math.sqrt(self.embed_size) # (tgt_len, b, embed_size)
        
        # Add learned positional embeddings
        src_pe = self.pos_embeddings(torch.arange(src_len, device=self.device).unsqueeze(1)) # (src_len, 1, embed_size)
        tgt_pe = self.pos_embeddings(torch.arange(tgt_len, device=self.device).unsqueeze(1)) # (tgt_len, 1, embed_size)

        source_embeddings = source_embeddings + src_pe # (src_len, b, embed_size)
        target_embeddings = target_embeddings + tgt_pe # (src_len, b, embed_size)

        source_embeddings = self.dropout(source_embeddings)
        target_embeddings = self.dropout(target_embeddings)

        # Generate Target and Pad masks
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len-1).to(self.device) # (tgt_len-1, tgt_len-1)
        
        src_pad_mask = (source_padded == self.vocab.src['<pad>']).transpose(0, 1) # (seq_len, b)
        tgt_pad_mask = (target_padded[:-1] == self.vocab.tgt['<pad>']).transpose(0, 1) # (tgt_len-1, b)

        # Generate outputs
        outputs = self.transformer(
            source_embeddings, # (src_len, b, embed_size)
            target_embeddings[:-1], # (tgt_len-1, b, embed_size)
            tgt_mask=tgt_mask, # (tgt_len-1, tgt_len-1)
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask
        ) # (tgt_len-1, b, embed_size)

        P = F.log_softmax(self.target_vocab_projection(outputs), dim=-1) # (tgt_len-1, b, embed_size)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        
        return scores

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[
        Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        memory = self.encode(src_sents_var)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        start_id = self.vocab.tgt['<s>']
        eos_id   = self.vocab.tgt['</s>']
        
        # hypotheses: list of (token_ids, score)
        hypotheses = [
            (torch.tensor([start_id], device=self.device), 0.0)
        ]
        completed_hypotheses = []
        
        t = 0
        while t < max_decoding_time_step and len(completed_hypotheses) < beam_size:
            t += 1
            new_hypotheses = []
        
            for tokens, score in hypotheses:
                # If already ended, keep it
                if tokens[-1].item() == eos_id:
                    completed_hypotheses.append(
                        Hypothesis(
                            value=self.vocab.tgt.indices2words(tokens.tolist()[1:-1]),
                            score=score
                        )
                    )
                    continue
        
                # Decode full prefix
                tgt = tokens.unsqueeze(1)              # Now (T, 1)
                decoder_out = self.decode(tgt, memory) # Now returns (T, 1, E)
            
                # CHANGE: Index the LAST time step (Dim 0)
                last_word_hidden = decoder_out[-1, 0, :] # Shape: (E,)
                logits = self.target_vocab_projection(last_word_hidden)
                log_probs = F.log_softmax(logits, dim=-1)
                """
                tgt = tokens.unsqueeze(0)              # (1, T)
                decoder_out = self.decode(tgt, memory) # (1, T, d_model)
        
                logits = self.target_vocab_projection(decoder_out[:, -1])
                log_probs = F.log_softmax(logits, dim=-1)
                """

                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
        
                # Iterate over the beam dimension (dim 1)
                for i in range(beam_size):
                    log_p = topk_log_probs[i]
                    idx = topk_ids[i]
                    
                    new_tokens = torch.cat([tokens, idx.unsqueeze(0)])
                    new_score = score + log_p.item()
                    new_hypotheses.append((new_tokens, new_score))
        
            # Prune
            hypotheses = sorted(
                new_hypotheses, key=lambda x: x[1], reverse=True
            )[:beam_size]


        if len(completed_hypotheses) == 0:
            tokens, score = hypotheses[0]
            completed_hypotheses.append(
                Hypothesis(
                    value=self.vocab.tgt.indices2words(tokens.tolist()[1:]),
                    score=score
                )
            )
        
        completed_hypotheses.sort(key=lambda h: h.score, reverse=True)
        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.source_layer.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(
                embed_size=self.embed_size,
                dropout_rate=self.dropout_rate,
                nhead=self.nhead,                            
                num_encoder_layers=self.num_encoder_layers,  
                num_decoder_layers=self.num_decoder_layers,  
                max_len=self.pos_embeddings.num_embeddings,   
            ),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
