from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """

    def __init__(self, embed_size, hidden_size, vocab, nhead=6, num_encoder_layers=3, num_decoder_layers=3, dropout_rate=0.1):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size, the size of hidden states (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.nhead=nhead
        self.num_encoder_layers=num_encoder_layers
        self.num_decoder_layers=num_decoder_layers

        # We are using nhead=8 because our embedding size is 200
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dropout=self.dropout_rate
        )

        self.target_vocab_projection=nn.Linear(embed_size, len(self.vocab.tgt))

    def encode(self, source_padded):
        # Use model embeddings to embed source tensors
        source_embeddings       = self.model_embeddings(source_padded, is_source=True)
        return self.transformer.encoder(source_embeddings)

    def decode(self, target_padded, memory):
        # Use model embeddings to embed source tensors
        target_embeddings       = self.model_embeddings(target_padded, is_source=False)

        target_mask = self.transformer.generate_square_subsequent_mask(target_embeddings.size(0)).to(target_padded.device)

        return self.transformer.decoder(target_embeddings, memory, tgt_mask=target_mask)
        

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

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)  # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)  # Tensor: (tgt_len, b)

        # Use model embeddings to embed source and target tensors
        source_embeddings       = self.model_embeddings(source_padded, is_source=True)
        target_embeddings       = self.model_embeddings(target_padded, is_source=False)

        # Generate Masks
        tgt_len = target_padded.size(0)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len-1).to(self.device)
        
        # Parallel Forward Pass
        src_pad_mask = (source_padded == self.vocab.src['<pad>']).transpose(0, 1)
        tgt_pad_mask = (target_padded[:-1] == self.vocab.tgt['<pad>']).transpose(0, 1)

        outputs = self.transformer(
            source_embeddings,
            target_embeddings[:-1],
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )

        P = F.log_softmax(self.target_vocab_projection(outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        
        return scores

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size.
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.

        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

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
                            value=self.vocab.tgt.to_words(tokens.tolist()[1:-1]),
                            score=score
                        )
                    )
                    continue
        
                # Decode full prefix
                tgt = tokens.unsqueeze(0)              # (1, T)
                decoder_out = self.decode(tgt, memory) # (1, T, d_model)
        
                logits = self.target_vocab_projection(decoder_out[:, -1])
                log_probs = F.log_softmax(logits, dim=-1)
        
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
        
                for log_p, idx in zip(topk_log_probs[0], topk_ids[0]):
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
                    value=self.vocab.tgt.to_words(tokens.tolist()[1:]),
                    score=score
                )
            )
        
        completed_hypotheses.sort(key=lambda h: h.score, reverse=True)
        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source_layer.weight.device

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
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
