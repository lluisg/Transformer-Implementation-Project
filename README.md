# SIMPLE TRANSFORMER IMPLEMENTATION PROJECT
# Motivation
The transformer is one of state of the art models in NLP and many other tasks. And so I was interested in knowing how it works. After some time spent in research and understanding, I decided that the next logical step would be to try to implement it in some simple task, and see how it goes. The task chosen for this was to fill the blank word in a sentence (the language selected was Spanish).

To do so I implemented a Colab notebook with everything, using fewer sentences despite knowing that it will obtain worse results to make it work more quickly and assure it works. Then I adapted it into a python file which ran with more sentences (see the table of characteristics below), more power and tries to obtain better results.

The Transformer implementation is based on the one from Samuel Lynn-Evans in [his blog](https://blog.floydhub.com/the-transformer-in-pytorch/).

# Dependencies:

- Python 3.7
- Torch
- numpy
- pandas
- nltk
- bs4
- wget
- sklearn
- matplotlib



### Data Used
For the data, I used the monolingual Spanish sets given by WMT20 for the translation task: Europarl v10 and News Crawl 2007-2019. The information about the number of sentences used as well the vocabulary size can be seen on the following table:

| Version | Total Sentences | After Preprocessing | Train | Valid | Test | #words vocabulary | Total words |
| ------ | ------ | ------ |  ------ | ------ | ------ | ------ | ------ |
| Colab | 10.000 | 8.395 |  6.294 | 1.259 | 840 | 5.000 | 20.827 |
| Extended | 2.000.000 | |

### Results

The results obtained were of:

| Version | Epochs | Training Loss | Valid Loss | Test Loss |
| ------ | ------ | ------ |  ------ | ------ |
| Colab | 30 | 2.76 |  11.63 | 1.61 |
| Extended | 50 | |

The loss was computed using Cross Entropy Loss between the last state of the result given by the Transformer and it's supposed value.

It was a little strange because in the reduced version, the validation loss increased from the very beginning despite the training loss decreasing on all epochs, I suppose it was provoked for the fact that the list of sentences is not as large as it should be, taking into account that it has to learn a 5.000 vocabulary with 8.000 sentences only.

