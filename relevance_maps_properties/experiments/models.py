import os
import torch

from typing import Any
import numpy as np

from collections import defaultdict
from dianna .utils.tokenizers import SpacyTokenizer
from skimage.segmentation import slic
from typing import Optional
from torch import nn
from torchtext.vocab import Vectors
import torch.nn.functional as F


class Slic_Wrapper():
    def __init__(self, 
                 n_segments: int = 10,
                 compactness: float = 10.,
                 sigma: float = 0.):
        self.n_segments= n_segments
        self.compactness = compactness
        self.sigma = sigma

    def __call__(self, image):
        return slic(image, 
                    n_segments=self.n_segments, 
                    compactness=self.compactness, 
                    sigma = self.sigma)
    
    def __repr__(self):
        return f'slic(n_segments={self.n_segments}, compactness={self.compactness}, sigma={self.sigma})'
    

class Model(nn.Module):
    '''Code from Leon Oostrum, this class is necessary to retrieve the relevant
       pytorch model. 
    '''
    def __init__(self, vocab_size, embedding_size, n_filters, filter_sizes, padding_idx,
                dropout, output_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        
        self.conv_layers = nn.ModuleList()
        for filter_size in filter_sizes:
            layer = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size, embedding_size))
            self.conv_layers.append(layer)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_sizes), output_dim)
    
    def forward(self, text):
        # shape = [batch size, max nword per sentence]
        embedding = self.embedding(text).unsqueeze(1)
        # shape = [batch_size, 1, nword, embedding dim]
        conved = [F.relu(conv(embedding)).squeeze(3) for conv in self.conv_layers]
        # shape = len(filter_sizes) list of [batch_size, n_filter, nword - filter_size + 1]
        # note: max_pool1d does not work with ONNX when output shape is dynamic
        # therefore switched to adaptive_max_pool1d
        pooled = [F.adaptive_max_pool1d(out, 1).squeeze(2) for out in conved]
        # shape = len(filter_sizes) list of [batch_size, n_filter]
        concat = torch.cat(pooled, dim=1)
        # shape = [batch_size * len(filter_sizes), n_filter]
        dropped = self.dropout(concat)
        return self.fc(dropped)
    

class Predictor:
    '''Again, from the work of Leon Oostrum, this is necessary for the model to work.
    '''
    def __init__(self, model, tokenizer, word_vectors, device, max_filter_size=5):
        self.model = model
        self.tokenizer = SpacyTokenizer(name='en_core_web_sm')
        self.vocab = Vectors(word_vectors, cache='.')
        self.device = device
        self.max_filter_size = max_filter_size

    def _preprocess_batch(self, sentences):
        batch = []
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            # pad if needed
            if self.max_filter_size is not None:
                tokens = self._pad(tokens, self.max_filter_size)
            # numericalize
            tokens = [self.vocab.stoi[token] if token in self.vocab.stoi 
                      else self.vocab.stoi['<unk>'] for token in tokens]
            batch.append(tokens)
        return batch

    def _pad(self, tokens, max_filter_size, padding_token='<pad>'):
        npad = max(max_filter_size - len(tokens), 0)
        tokens += [padding_token] * npad
        return tokens
            
    def __call__(self, sentences, preprocess=True):
        # Handle preprocessing
        if isinstance(sentences[0][0], str):
            batch = self._preprocess_batch(sentences)
        else:
            batch = sentences
            
        # move to device and add required batch axis
        tokens = torch.tensor(batch).to(self.device)
        # feed to model
        pred = torch.sigmoid(self.model.forward(tokens)).cpu().detach().numpy()
        return np.hstack((pred, 1-pred))