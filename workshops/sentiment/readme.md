# Sentiment analysis

## Useful links

[Practical guide to undocumented features](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/)

## Twitter dataset

Twitter sentiment analysis dataset cna be downloaded from [kaggle](https://www.kaggle.com/kazanova/sentiment140).

For more details refer to the [paper](https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf).

Data consists of the following columns:

|target|id|date|flag|user|text|
|---|---|---|---|---|---|
|0 negative; 2 neutral; 4 positive|---|---|---|---|---|

## Task 1. Create datasets (train, valid, test)

How do we split data into training, validation and test sets?

## Task 2. Explore target sentiment

How many different sentiments are present in tweet corpus?

Count occurrences of each target sentiment.

## Task 3. Create dictionary

How many different words appear in tweet corpus?

Count occurrences of each word in dictionary.

How to reduce number of words in dictionary?

What's the trade-off between dictionary size and tweet corpus coverage?

Hint: Take a look at tweet tokenization from ```nltk``` library.

## Task 4. Word embeddings

Implement word embeddings.

## Task 5. Recurrent neural networks

Implement vanilla RNN layer on top of word embeddings.

How to deal with exploding gradients?

## Task 6. Loss and accuracy

Implement loss and accuracy nodes.

## Task 7. Training speed

How much time is needed for a single training epoch?

How to speed-up the training?

What is potential problem with padding?

Which mechanism should be used so that padding doesn't affect results.

## Results

### Experiment 0. Vanilla RNN

Hyper-parameters:
* Batch size = 1024
* Word embedding = 100
* RNN units = 32

```
INFO:root:train lasted 00:07:07 acc=0.82 loss=0.39
INFO:root:valid lasted 00:00:31 acc=0.81 loss=0.41
```

### Experiment 1. GRU RNN

Hyper-parameters:
* Batch size = 1
* Word embedding = 100
* RNN units = 32

```
INFO:root:train lasted 07:15:42 acc=0.81 loss=0.42
eval |##################################################| ETA 00:00:00 (1280K) 100.0%
INFO:root:eval lasted 00:31:45
INFO:root:[epoch 0] train acc 0.8281238043928072
valid |##################################################| acc=0.82 ETA 00:00:00 (159K) 100.0%
INFO:root:valid lasted 00:05:03 acc=0.82 loss=0.39
eval |##################################################| ETA 00:00:00 (159K) 100.0%
INFO:root:eval lasted 00:04:09
INFO:root:[epoch 0] valid acc 0.8229330733268332
```

### Experiment 2. GRU RNN

Hyper-parameters:
* Batch size = 1
* Word embedding = 300
* RNN units = 128

```
INFO:root:train lasted 09:04:04 acc=0.82 loss=0.41
eval |##################################################| ETA 00:00:00 (1280K) 100.0%
INFO:root:eval lasted 03:03:44
INFO:root:[epoch 0] train acc 0.8331396937683977
valid |##################################################| acc=0.83 ETA 00:00:00 (159K) 100.0%
INFO:root:valid lasted 00:25:12 acc=0.83 loss=0.39
```

## Future work

### Evaluate model

Implement model evaluation on test set.

### Implement LSTM layer

What is different compared to vanilla RNN?

### Implement bidirectional layer

What is different compared to uni-directional vanilla RNN?

### TensorFlow data set

Explore support for batching and padding in TensorFlow.

Is there an option to prefetch data in parallel with training?

### IMDB review dataset

Try to solve the same problem for IMDB review dataset at [kaggle](https://www.kaggle.com/utathya/imdb-review-dataset)
