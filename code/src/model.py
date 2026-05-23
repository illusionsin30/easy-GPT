import math
import torch
import torch.nn as nn


class CausalLMM(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a stack of Transformer blocks and an output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    # The Transformer blocks have input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, vocab_size, dim=256, num_layers=4, num_heads=8):
        super(CausalLMM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim)

        # Construct you Transformer model

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        # Init model weights
        pass

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        # Write code here

        x = self.norm(x)
        x = self.head(x)
        return x
