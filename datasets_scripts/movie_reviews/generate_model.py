#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import tqdm
import wandb
import argparse

from torchtext.vocab import Vectors
from torchtext.data import get_tokenizer

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F


# class to hold a dataset
# custom datasets need to implement init, len, and getitem
class MovieReviewDataset(Dataset):
    def __init__(self, filename, tokenizer, vocab, max_samples=None, max_filter_size=None):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_filter_size = max_filter_size

        self.data = pd.read_csv(filename, delimiter='\t')
        if max_samples is not None:
            self.data = self.data[:min(len(self.data), max_samples)]

    def __getitem__(self, idx):
        # get sentence and corresponding label
        sentence = self.data.iloc[idx]['sentence']
        label = self.data.iloc[idx]['label']
        # tokenize sentence
        tokens = self.tokenizer(sentence)
        # pad if needed
        if self.max_filter_size is not None:
            tokens = pad(tokens, self.max_filter_size)
        # numericalize
        tokens_numerical = [self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['<unk>'] for token in tokens]
        return tokens_numerical, label

    def __len__(self):
        return len(self.data)


# function to pad to some minimum length
def pad(tokens, max_filter_size, padding_token='<pad>'):
    npad = max(max_filter_size - len(tokens), 0)
    tokens += [padding_token] * npad
    return tokens


# custom collate function to handle variable-size input
def collate(batch, padding_idx):
    # get max sample size: item[0] is the input sentence split into numerical tokens
    tokens = [sample[0] for sample in batch]
    max_size = max(map(len, tokens))
    # create input array with padding such that each element has the same size
    input_data = np.empty((len(batch), max_size), dtype=int)
    for i, item in enumerate(tokens):
        # pad each element and store
        input_data[i] = pad(item, max_size, padding_token=padding_idx)

    # convert to tensors
    input_data = torch.LongTensor(input_data)
    labels = torch.FloatTensor([item[1] for item in batch])
    return [input_data, labels]


# the model
class Model(nn.Module):
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


# function to calculate accuracy
def accuracy(model_output, y_true):
    y_pred = torch.round(torch.sigmoid(model_output))
    return (y_pred == y_true).sum() / len(y_pred)


def train(model, train_data, optimizer, loss_func):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in tqdm.tqdm(train_data):
        input_data, label = batch
        input_data = input_data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        predictions = model(input_data).squeeze(1)

        loss = loss_func(predictions, label)
        acc = accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        wandb.log({'train_loss': loss.item(), 'train_acc': acc.item()})

    epoch_loss /= len(train_data)
    epoch_acc /= len(train_data)

    wandb.log({'train_epoch_loss': epoch_loss, 'train_epoch_acc': epoch_acc})
    return epoch_loss, epoch_acc


def evaluate(model, data, loss_func):
    loss = 0
    acc = 0
    model.eval()

    with torch.no_grad():
        for batch in data:
            input_data, label = batch
            input_data = input_data.to(device)
            label = label.to(device)
            predictions = model(input_data).squeeze(1)

            loss += loss_func(predictions, label).item()
            acc += accuracy(predictions, label).item()

    loss /= len(data)
    acc /= len(data)

    return loss, acc


# class to predict sentiment from a sentence
class Predictor:
    def __init__(self, model, tokenizer, vocab, device, max_filter_size=None):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.device = device
        self.max_filter_size = max_filter_size
        self.classes = ['negative', 'positive']

        self.model.eval()

    def __call__(self, sentence):
        # get numerical tokens
        tokens = tokenizer(sentence)
        # pad if needed
        if self.max_filter_size is not None:
            tokens = pad(tokens, self.max_filter_size)
        # numericalize
        tokens = [self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['<unk>'] for token in tokens]
        # move to device and add required batch axis
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)
        # feed to model
        pred = torch.sigmoid(model.forward(tokens)).item()
        # get string representation of predicted class
        class_str = self.classes[int(np.round(pred))]

        return pred, class_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--n_filters', type=int)
    # parser.add_argument('--filter_sizes')
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--nosweep', action='store_true')
    parser.add_argument('--nowandb', action='store_true')

    args = parser.parse_args()

    config = wandb.config
    for key, value in vars(args).items():
        setattr(config, key, value)

    config.filter_sizes = [3, 4, 5]
    config.output_dim = 1
    config.max_samples = 10000

    if args.nowandb:
        mode = 'disabled'
    else:
        mode = 'online'
    wandb.init(project='movie-reviews', entity='dianna-ai', mode=mode)

    # select device to run on
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'PyTorch will use {device}')

    # path to data files
    data_path = os.path.join(os.path.expanduser('~'), 'surfdrive/Shared/datasets/stanford_sentiment_treebank_v2')

    # tokenizer for splitting input sentences
    tokenizer = get_tokenizer('spacy', 'en_core_web_sm')

    # word vectors as generated from data preparation notebook
    vocab = Vectors('word_vectors.txt')

    # convert collate function to callable that only requires batch as argument
    collate_func = lambda batch: collate(batch, vocab.stoi['<pad>'])

    # Load data into PyTorch dataset
    # These datasets return the input as numerical values, suited for input to the model
    train_data = MovieReviewDataset(os.path.join(data_path, 'train.tsv'), tokenizer, vocab, max_samples=config.max_samples)
    val_data = MovieReviewDataset(os.path.join(data_path, 'validation.tsv'), tokenizer, vocab)

    batch_size = config.batch_size
    train_data_iterator = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_func)
    val_data_iterator = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=collate_func)

    # initialize model
    output_dim = config.output_dim
    dropout = config.dropout
    n_filters = config.n_filters
    filter_sizes = config.filter_sizes

    vocab_size, embedding_size = vocab.vectors.size()
    padding_idx = vocab.stoi['<pad>']

    model = Model(vocab_size, embedding_size, n_filters, filter_sizes, padding_idx, dropout, output_dim)
    # copy pre-trained embeddings into model
    model.embedding.weight.data.copy_(vocab.vectors)

    model = model.to(device)

    wandb.watch(model)

    # define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_func = nn.BCEWithLogitsLoss().to(device)

    # do the training
    best_val_loss = np.inf
    best_val_acc = np.inf

    for epoch in range(config.epochs):
        train_epoch_loss, train_epoch_acc = train(model, train_data_iterator, optimizer, loss_func)
        val_epoch_loss, val_epoch_acc = evaluate(model, val_data_iterator, loss_func)
        # log the validation results to wandb
        wandb.log({'val_epoch_loss': val_epoch_loss, 'val_epoch_acc': val_epoch_acc})
        # only print some info and store model if not doing a wandb sweep
        if not args.nosweep:
            continue
        print(f'train loss: {train_epoch_loss:.2f} | train acc: {train_epoch_acc:.2f}')
        print(f'val   loss: {val_epoch_loss:.2f} | val   acc: {val_epoch_acc:.2f}')
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_val_acc = val_epoch_acc
            # ensure we are in eval mode
            model.eval()
            torch.save(model, 'movie_review_model.pytorch')
    
    # stop here if doing a wandb sweep
    if not args.nosweep:
        exit()

    print(f"Best validation loss: {best_val_loss:.2f}, accuracy: {best_val_acc:.2f}")

    # load best model from disk
    loaded_model = torch.load('movie_review_model.pytorch')
    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    # store as ONNX, needs example input
    x = next(iter(train_data_iterator))[0].to(device)
    torch.onnx.export(loaded_model, x, 'movie_review_model.onnx', opset_version=11)

    # create predictor
    predict_sentiment = Predictor(loaded_model, tokenizer, vocab, device, max_filter_size=max(config.filter_sizes))

    # print some predictions from the (unlabeled) test set
    sentences = pd.read_csv(os.path.join(data_path, 'test.tsv'), delimiter='\t')['sentence']
    nmax = 10
    classes = ['negative', 'positive']

    for n, sentence in enumerate(sentences):
        if n == nmax:
            break
        output_numerical, predicted_class = predict_sentiment(sentence)
        print(f"\"{sentence}\" - {predicted_class} - {output_numerical:.2f}")
