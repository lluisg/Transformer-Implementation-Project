
import io
import sys
import os
from os import path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

import torch
import torch.optim as optim
import torch.nn as nn

import wget
import shutil
import gzip
import random
import copy
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

from transformer_model import Transformer

def sentence_to_words(sentence):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()

    text = BeautifulSoup(sentence, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("spanish")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem

    return words

def build_dict(data, vocab_size = 5000):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""

    # TODO: Determine how often each word appears in `data`. Note that `data` is a list of sentences and that a
    #       sentence is a list of words.
    word_count = {} # A dict storing the words that appear in the reviews along with how often they occur

    for sentence in data:
        for word in sentence:
            if word in word_count:
                word_count[word]+=1
            else:
                word_count[word]=1

    # TODO: Sort the words found in `data` so that sorted_words[0] is the most frequently appearing word and
    #       sorted_words[-1] is the least frequently appearing word.

    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    sorted_words = [tupl[0] for tupl in sorted_words]

    word_dict = {} # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 3]): # The -3 is so that we save room for the 'no word'
        word_dict[word] = idx + 3                              # 'infrequent' 'pad' labels

    return word_dict, word_count

def convert_and_pad(word_dict, sentence, pad=500):
    NOWORD = 1 # We will use 0 to represent the 'no word' category
    INFREQ = 2 # and we use 1 to represent the infrequent words, i.e., words not appearing in word_dict

    working_sentence = [NOWORD] * pad

    for word_index, word in enumerate(sentence[:pad]):
      if word_index < pad:
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ

    return working_sentence, min(len(sentence), pad)

def convert_and_pad_data(word_dict, data, pad=500):
    result = []
    lengths = []

    for sentence in data:
        converted, leng = convert_and_pad(word_dict, sentence, pad)
        result.append(converted)
        lengths.append(leng)

    return np.array(result), np.array(lengths)

def create_mask(data, batchsize):
  input_msk = [[[0] if _el == 1 else [1] for _el in _ar] for _ar in data]
  arr_input_msk = np.array(input_msk)

  torch_msk_input = torch.tensor(input_msk).clone()
  mask_sample_ds = torch.utils.data.TensorDataset(torch_msk_input)
  msk_input_loader = torch.utils.data.DataLoader(mask_sample_ds, batch_size=batchsize)

  return msk_input_loader


def train(model, train_loader, mask_loader, eval_loader, eval_mask_loader, epochs, optimizer, loss_fn, device):
    print('Start training')
    sys.stdout.flush()
    total_length = len(train_loader.dataset)
    loss_return = []
    eval_loss_return = []

    for epoch in range(1, epochs + 1):
        model.train()
        batchs_done = 0
        total_loss = 0

        for batch, msk_batch in zip(train_loader, mask_loader):
            batch_X, batch_y = batch
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            if len(msk_batch)==2:
              input_msk, output_msk = msk_batch
              input_msk = input_msk.to(device)
              output_msk = output_msk.to(device)
            else:
              input_msk = msk_batch[0]
              output_msk = None
              input_msk = input_msk.to(device)

            out = model(batch_X, batch_X, input_msk, output_msk)

            optimizer.zero_grad()

            batch_loss = loss_fn(out[:, -1, :], batch_y)

            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.data.item()

        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))
        loss_return.append(total_loss / len(train_loader))
        sys.stdout.flush()

        if epoch % 5 == 0: #every five epochs an evauation
          eval_total_loss = 0
          model.eval()

          for eval_batch, eval_msk_batch in zip(eval_loader, eval_mask_loader):
            eval_batch_X, eval_batch_y = eval_batch
            eval_batch_X = eval_batch_X.to(device)
            eval_batch_y = eval_batch_y.to(device)

            if len(eval_msk_batch)==2:
              eval_input_msk, eval_output_msk = eval_msk_batch
              eval_input_msk = eval_input_msk.to(device)
              eval_output_msk = eval_output_msk.to(device)
            else:
              eval_input_msk = eval_msk_batch[0]
              eval_output_msk = None
              eval_input_msk = eval_input_msk.to(device)

            eval_out = model(eval_batch_X, eval_batch_X, eval_input_msk, eval_output_msk)

            eval_batch_loss = loss_fn(eval_out[:, -1, :], eval_batch_y)
            eval_total_loss += eval_batch_loss.data.item()

          print("Eval Loss: {}".format(eval_total_loss / len(eval_loader)))
          eval_loss_return.append(eval_total_loss / len(eval_loader))
          sys.stdout.flush()

    return loss_return, eval_loss_return

def evaluate(model, test_loader, masks_loader, loss_fn, batch_size, device):
  print('Start evaluating')
  total_loss = 0
  model.eval()

  for batch, msk_batch in zip(test_loader, masks_loader):
        batch_X, batch_y = batch
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        if len(msk_batch)==2:
          input_msk, output_msk = msk_batch
          input_msk = input_msk.to(device)
          output_msk = output_msk.to(device)
        else:
          input_msk = msk_batch[0]
          output_msk = None
          input_msk = input_msk.to(device)

        len_batch = len(batch_X)

        out = model(batch_X, batch_X, input_msk, output_msk)

        batch_loss = loss_fn(out[:, -1, :], batch_y)
        total_loss += batch_loss.data.item()

  return total_loss / len(train_loader)

def guess_word(custom_sentence):
  # PREPROCESS
  custom_tok = sentence_to_words(custom_sentence)

  # LIMIT LENGTH
  if len(custom_tok) > MAX_PADDING:
    custom_tok = custom_tok[:MAX_PADDING]

  # CONVERT AND PAD
  custom_int, custom_len = convert_and_pad(word_dict, custom_tok, MAX_PADDING)

  # MASK
  msk_input_custom = [[0] if _el == 1 else [1] for _el in custom_int]
  torch_msk_input = torch.tensor(msk_input_custom).clone().unsqueeze(0).to(device)
  output_mask = None


  #PASS THROUGH THE MODEL
  torch_custom = torch.tensor(custom_int).clone().unsqueeze(0).to(device)
  model.eval()
  custom_out = model(torch_custom, torch_custom, torch_msk_input, output_mask)

  custom_out = custom_out[:,-1,:].cpu().detach().numpy().reshape(-1)
  ind_max = np.argmax(custom_out, axis=0)

  for key, value in word_dict.items():
    if ind_max == value:
      resulting_word = key

  return resulting_word


# MAIN -------------------------------------------------------------------------
# --------------------------------------------------------------------------
def main(DATA, MAX_LINES, MAX_PADDING, MIN_LEN_SENTENCE, SIZE_VOCAB, SHOW_SENTENCES, LR, EPOCHS):

    if DATA == 'europarl':
        URLS=["http://www.statmt.org/europarl/v10/training-monolingual/europarl-v10.es.tsv.gz"]
        FILES = ["europarl-v10.es.tsv.gz"]
        CORPORA = ["europarl-v10.es.tsv"]

    elif DATA == 'newscarl':
        URLS=[
        "http://data.statmt.org/news-crawl/es/news.2007.es.shuffled.deduped.gz",
        "http://data.statmt.org/news-crawl/es/news.2008.es.shuffled.deduped.gz",
        "http://data.statmt.org/news-crawl/es/news.2009.es.shuffled.deduped.gz",
        "http://data.statmt.org/news-crawl/es/news.2010.es.shuffled.deduped.gz",
        "http://data.statmt.org/news-crawl/es/news.2011.es.shuffled.deduped.gz",
        "http://data.statmt.org/news-crawl/es/news.2012.es.shuffled.deduped.gz",
        "http://data.statmt.org/news-crawl/es/news.2013.es.shuffled.deduped.gz",
        "http://data.statmt.org/news-crawl/es/news.2014.es.shuffled.deduped.gz",
        "http://data.statmt.org/news-crawl/es/news.2015.es.shuffled.deduped.gz",
        "http://data.statmt.org/news-crawl/es/news.2016.es.shuffled.deduped.gz",
        "http://data.statmt.org/news-crawl/es/news.2017.es.shuffled.deduped.gz",
        "http://data.statmt.org/news-crawl/es/news.2018.es.shuffled.deduped.gz",
        "http://data.statmt.org/news-crawl/es/news.2019.es.shuffled.deduped.gz"
      ]

        FILES=[
          "news.2007.es.shuffled.deduped.gz",
          "news.2008.es.shuffled.deduped.gz",
          "news.2009.es.shuffled.deduped.gz",
          "news.2010.es.shuffled.deduped.gz",
          "news.2011.es.shuffled.deduped.gz",
          "news.2012.es.shuffled.deduped.gz",
          "news.2013.es.shuffled.deduped.gz",
          "news.2014.es.shuffled.deduped.gz",
          "news.2015.es.shuffled.deduped.gz",
          "news.2016.es.shuffled.deduped.gz",
          "news.2017.es.shuffled.deduped.gz",
          "news.2018.es.shuffled.deduped.gz",
          "news.2019.es.shuffled.deduped.gz"
        ]

        CORPORA=[
          "news.2007.es.shuffled.deduped",
          "news.2008.es.shuffled.deduped",
          "news.2009.es.shuffled.deduped",
          "news.2010.es.shuffled.deduped",
          "news.2011.es.shuffled.deduped",
          "news.2012.es.shuffled.deduped",
          "news.2013.es.shuffled.deduped",
          "news.2014.es.shuffled.deduped",
          "news.2015.es.shuffled.deduped",
          "news.2016.es.shuffled.deduped",
          "news.2017.es.shuffled.deduped",
          "news.2018.es.shuffled.deduped",
          "news.2019.es.shuffled.deduped"
        ]

    print('File download') #---------------------------------------------
    for u, f in zip(URLS, FILES):
        print(u)
        sys.stdout.flush()
        if path.exists(f):
            print('File already downloaded'.format(f))
        else:
            wget.download(u, './'+f)


    print('Unzipping {}'.format(f)) #----------------------------------------------
    for f, c in zip(FILES, CORPORA):
        print(f)
        sys.stdout.flush()
        if path.exists(f):
            with gzip.open(f, 'rb') as f_in:
                with open(c, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print('File already unzipped')


    print('Join all into one') #---------------------------------------------------
    sys.stdout.flush()
    with open('corpus.es', 'wb') as outfile:
        for fname in CORPORA:
            print('Joining file:', fname)
            sys.stdout.flush()
            with open(fname, 'rb') as infile:
                for line in infile:
                    outfile.write(line)


    print('Delete auxiliar files') #-----------------------------------------------
    sys.stdout.flush()
    for f in CORPORA:
        os.remove(f)

    print('Reduced File') #--------------------------------------------------------
    sys.stdout.flush()

    with open('corpus.es', 'rb') as cfile:
      with open('corpus.reduced.es', 'wb') as rfile:
        count = 0
        while count < MAX_LINES:
          line = cfile.readline()
          rfile.write(line)
          count += 1
        print('Number of lines in the reduced file:', count)


    print('Read Data') #----------------------------------------------------------
    sys.stdout.flush()
    FILE = 'corpus.reduced.es'
    data = []

    with open(FILE, 'rb') as corpus_file:
      Lines = corpus_file.readlines()
      for line in Lines:
        data.append(line)


    print('Preprocessing the Data') #----------------------------------------------
    sys.stdout.flush()

    print(data[50])
    print(sentence_to_words(data[50]))
    sys.stdout.flush()

    preprocessed_data = []
    perc = 0

    for ind, s in enumerate(data):
      words = sentence_to_words(s)
      if len(words) >= MIN_LEN_SENTENCE and len(words) <= MAX_PADDING :
        preprocessed_data.append(words)
      if ind > perc:
        print('{}/{} sentences preprocessed'.format(perc, len(data)))
        sys.stdout.flush()
        perc += SHOW_SENTENCES

    print('Length data readed: ', len(data))
    print('Length after preprocessing: ', len(preprocessed_data))
    print(preprocessed_data[50])
    sys.stdout.flush()


    print('Build Dictionary') #---------------------------------------------------
    sys.stdout.flush()

    FED = 20 #nmber of elements we will show
    word_dict, complete_dict = build_dict(preprocessed_data, SIZE_VOCAB)
    list_dict = [key for key in word_dict.keys()]
    print('First {} elements of the dictionary: {}'.format(FED, list_dict[:FED]))
    sys.stdout.flush()

    print('Selected {}/{} words'.format(SIZE_VOCAB, len(complete_dict)))
    sys.stdout.flush()

    # We will add the XXX as 0 in the dictionary, so when a custom sentence is inputted, the XXX will mark the word the model has to guess
    word_dict['XXX'] = 0
    word_dict['PAD'] = 1
    word_dict['INFREQ'] = 2


    print('Convert and Pad') #-----------------------------------------------------
    sys.stdout.flush()

    int_data, int_data_len = convert_and_pad_data(word_dict, preprocessed_data, MAX_PADDING)
    print(int_data[50], int_data_len[50])
    sys.stdout.flush()


    print('Extract Word') #-------------------------------------------------------
    sys.stdout.flush()

    # Check there is no sentence with all 2's
    int_data_pre = [d for d, lend in zip(int_data, int_data_len) if len(set(d[:lend])) > 1]
    len_data_pre = [lend for d, lend in zip(int_data, int_data_len) if len(set(d[:lend])) > 1]
    print('{} of the {} sentences were only 2\'s'.format(len(int_data)-len(int_data_pre), len(int_data)))
    sys.stdout.flush()

    masked_data = []
    word_masked = []

    masked_data = int_data_pre.copy()

    for idx, (sentence, len_sentence) in enumerate(zip(int_data_pre, len_data_pre)):
      acceptable_value = False

      while acceptable_value == False:
        idx_word = random.randint(0, len_sentence-1)
        if int_data_pre[idx][idx_word] != 2:
          acceptable_value = True

      word_masked.append(int_data_pre[idx][idx_word]) #save the word extracted
      masked_data[idx][idx_word] = 0 #put this word to 0

    print(masked_data[50])
    print(word_masked[50])
    sys.stdout.flush()


    print('Split Train, Valid and Test') #-----------------------------------------
    sys.stdout.flush()

    train_x, valid_x, train_y, valid_y, train_len, valid_len = train_test_split(masked_data, word_masked, len_data_pre, test_size=0.25, random_state=42)
    valid_x, test_x, valid_y, test_y, valid_len, test_len = train_test_split(valid_x, valid_y, valid_len, test_size=0.4, random_state=42)

    print('train: ', len(train_x), len(train_y), len(train_len))
    print('valid: ', len(valid_x), len(valid_y), len(valid_len))
    print('test: ', len(test_x), len(test_y), len(test_len))
    sys.stdout.flush()


    print('Cleaning Variables') #--------------------------------------------------
    sys.stdout.flush()
    preprocessed_data = None
    list_dict = None
    int_data = None
    int_data_len = None
    int_data_pre = None
    len_data_pre = None
    masked_data = None
    word_masked = None
    data = None
    Lines = None
    complete_dict = None


    # Training ------------------------------------------------------------------
    BATCH_SIZE = 128
    d_model = 256
    heads = 8
    N = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    sys.stdout.flush()

    print('Preparing Input Masks')
    sys.stdout.flush()
    msk_input_loader = create_mask(train_x, BATCH_SIZE)
    valid_msk_input_loader = create_mask(valid_x, BATCH_SIZE)

    print('Preparing Train Data Loaders')
    sys.stdout.flush()
    train_torch_x = torch.tensor(train_x).clone()
    train_torch_y = torch.tensor(train_y).clone()
    train_sample_ds = torch.utils.data.TensorDataset(train_torch_x, train_torch_y)
    train_loader = torch.utils.data.DataLoader(train_sample_ds, batch_size=BATCH_SIZE)

    print('Preparing Validation Data Loaders')
    sys.stdout.flush()
    valid_torch_x = torch.tensor(valid_x).clone()
    valid_torch_y = torch.tensor(valid_y).clone()
    valid_sample_ds = torch.utils.data.TensorDataset(valid_torch_x, valid_torch_y)
    valid_loader = torch.utils.data.DataLoader(valid_sample_ds, batch_size=BATCH_SIZE)

    print('Initialize Model')
    sys.stdout.flush()
    model = Transformer(SIZE_VOCAB, d_model, N, heads).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    print('Define Loss and Optimizer')
    sys.stdout.flush()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

    mask_sample_ds = None
    train_torch_x = None
    train_torch_y = None
    train_sample_ds = None

    print('Going to train')
    sys.stdout.flush()
    losses, valid_losses = train(model, train_loader, msk_input_loader, valid_loader, valid_msk_input_loader, EPOCHS, optimizer, loss_function, device)


    # Testing ------------------------------------------------------------------
    print('Preparing Input Masks')
    msk_test_input_loader = create_mask(test_x, BATCH_SIZE)

    print('Preparing Data Loaders')
    test_torch_x = torch.tensor(test_x).clone()
    test_torch_y = torch.tensor(test_y).clone()
    test_sample_ds = torch.utils.data.TensorDataset(test_torch_x, test_torch_y)
    test_loader = torch.utils.data.DataLoader(test_sample_ds, batch_size=BATCH_SIZE)

    test_torch_x = None
    test_torch_y = None
    test_sample_ds = None

    print('Going to test')
    test_loss = evaluate(model, test_loader, msk_test_input_loader, loss_function, BATCH_SIZE, device)
    print(test_loss)


    # Custom Sentence ------------------------------------------------------------
    test_sentences = ["Ha habido una XXX en Colombia durant la presentación del presidente",
          "Todas las tropas han sido XXX a America",
          "Estaba pensando que quizas XXX deberías hacerlo",
          "Todo lo que llevo esta dentro de mí XXX"]

    for custom_sentence in test_sentences:
      resulting_word = guess_word(custom_sentence)

      print('Initial Sentence: \t {}'.format(custom_sentence))
      print('Word Guessed: \t\t {}'.format(resulting_word))


if __name__ == "__main__":
    # DATA = 'europarl'
    DATA = 'newscarl'

    MAX_LINES = 2000000 #number of total lines used from the files
    MAX_PADDING = 35 #max length of the sentences allowed
    MIN_LEN_SENTENCE = 5 #min length of the sentences allowed
    SIZE_VOCAB = 20000 #size of the vocabulary used
    SHOW_SENTENCES = 100000 #increase of sentences to show when preprocessing

    LR = 0.001 #learning rate for training the model
    EPOCHS = 50 #number of epochs in training

    main(DATA, MAX_LINES, MAX_PADDING, MIN_LEN_SENTENCE, SIZE_VOCAB, SHOW_SENTENCES, LR, EPOCHS)
