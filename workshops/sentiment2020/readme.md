# RNN and NLP workshop

The main goal of this workshop is to give an overview of the basic concepts of **natural language processing (NLP)** and **recurrent neural networks (RNNs)**. As it introduces both RNNs and NLP, the main emphasis is on the implementation. The **sentiment classification** of tweets is taken as an example task to demonstrate the end-to-end implementation of the machine learning models commonly used natural language processing.

Understanding the supervised learning paradigm, foundational components of neural networks and basics of *PyTorch* is needed to follow this workshop.

It is important to note that the material shown in this workshop has purely demonstrative purpose, which is why some steps might be simplified or omitted. The examples include reducing the size of the dataset to speed-up the training or lack of hyperparameter tuning. It is strongly recommended to exercise and modify the code, to explore other options, starting from the choice of hyperparameters to the models' architecure.

## 1. Neccesary installs

The list of *Python* packages used in this workshop is the following:

1) **pandas** - *conda install pandas*
2) **nltk** - *pip install nltk*
3) **matplotlib** - *pip install matplotlib*
4) **sklearn** - *pip install scikit-learn*
5) **annoy** - *pip install annoy*
6) **torch** - *conda install pytorch torchvision -c pytorch*

## 2. Table of Contents

The notebooks for this workshop should be used in the following order:

1) **SentimentAnalysisIntrodiction.ipynb** introduces the sentiment analysis problem that is going to be solved and the *sentiment140* dataset used for this task. It covers the basic preprocessing done on the input data (class **TextPreprocessor**) and dataset split into training, validation and test partition (class **SplitDataset**).
<br/>
2) **SentimentAnalysisTraining.ipynb** notebook illustrates how to train a simple model for sentiment classification of tweets. It covers:
   - how to represent textual data in the numerical format (classes **Vocabulary**, **TwitterVectorizer** and its subclass **TwitterOneHotVectorizer**),
   - how to reprepresent dataset in that format (class **TwitterDataset**),
   - how to load that data for training (method in **TwitterDataLoader**)
   - how to build a simple perceptron model for sentiment classification (class **ModelPerceptron**)
   - how to implement the training loop that uses previously defined components: dataset, model, loss function and optimizer

    This notebook is the base for other steps in the workshop and it is important to understand it well before proceeding further with the workshop.
<br/>
3) **SentimentAnalysisEvaluation.ipynb** notebook tackles with the evaluation of the sentiment classification model built in the previous step (SentimentAnalysisTraining.ipynb notebook). It repeats steps introduced in the previous notebook: definition of dataset, model, loss function and optimizer, and encapsulates training loop into the separate class (class **Trainer**). It further covers the evaluation of the trained model, including evaluation on some held-out portion of the data (test set), inference on a new data or inspecting the model weights to see what it has learned. The setting used in this notebook (initialization, model training, evaluation) is used in all further examples during the workshop.
<br/>
4) [*Homework*] **SentimentAnalysisMLP.ipynb** notebook showcases what happens if the simple perception model is replaced with more complex model (class **ModelMLP**). Since the only difference is in model architecture, this example will not be covered during the workshop and is recommended as a homework task.
<br/>
5) **WordEmbeddings.ipynb** notebook introduces the concepts of word embeddings and demonstrates how to use pretrained word embeddings like GloVe (class **PreTrainedEmbeddings**). The analogy task is chosen to showcase some properties of word embeddings. This notebook concludes with the introduction of a new method of representing words in numerical format (classes **SequenceVocabulary** and **TwitterSequenceVectorizer**), that is tested on a few examples.
<br/>
6) **SentimentAnalysisMLPWithEmbeddings.ipynb** notebook explains how to use word embedding representation as an input to the multi-layer perceptron network for sentiment classification. It introduces the <code>nn.Embedding</code> layer, a PyTorch module that encapsulates an embedding matrix. That is followed by the description of the sentiment classification model (class **ModelMLPWithEmbeddings**), that covers two possible initializations of the <code>Embedding</code> layer: with and without pre-trained embedding vectors as initial values. During the workshop, the case with pre-trained GloVe embeddings as initial values will be covered, while the case when word embeddings are learnt from scratch is recommended as a homework task. The rest of the notebook has standard flow: initialization of dataset, model, loss function and optimizer, model training and finally, evaluation.
<br/>
7) **SentimentAnalysisElmanRNN.ipynb** notebook concludes this workshop with the description of recurrent neural networks and how they can be applied for the sentiment classification task. Firstly, it depicts the architecture of the *Elman RNN* module, that represents the base (vanilla) version of the RNN (class **ElmanRNN**). Further, it is showcased how to combine <code>Embedding</code> layer, the <code>ElmanRNN</code> module and <code>Linear</code> layers into a model for sentiment classification of tweets (class **ModelElmanRNN**). While this workshop covers only the architecture of the *Elman RNN*, it is strongly recommended to explore what happens if it is replaced with some other RNN module such as *GRU* or *LSTM*, as a homework task. The rest of the notebook has standard flow: initialization of dataset, model, loss function and optimizer, model training and finally, evaluation.

## 3. Tasks

This workshop has several tasks, to fill-in the missing code. The tasks are:

1) Implement training and validation loop steps - notebook **SentimentAnalysisTraining.ipynb**
<br/>
2) Implement <code>get_words_closest_to_vector(vector, num_words)</code> method that returns the words nearest to the given vector (class **PreTrainedEmbeddings**). Apply this method for the anology task, explained in the **WordEmbeddings.ipynb** notebook.
<br/>
3) Implement <code>make_embeddings_matrix(word_list)</code> method (class **PreTrainedEmbeddings**). It should return the embedding matrix with the embedding vectors for the words in the given list.
<br/>
4) Implement <code>vectorize(text, vector_length)</code> method in the class **TwitterSequenceVectorizer** that should create a vector for the given text of given length. Test this method on some examples in **WordEmbeddings.ipynb** notebook.
<br/>
5) Implement <code>forward(x, apply_softmax)</code> method in the class **ModelMLPWithEmbeddings**. It should execute the forward pass for the classifier model based on the multi-layer perceptron network.
<br/>
6) Implement <code>forward(x, apply_softmax)</code> method in the class **ModelElmanRNN**. It should execute the forward pass for the classifier model based on the Elman recurrent neural network.

## 4. References

1) Andrew Ng, "Deep Learning Specialization", Coursera

2) Delip Rao & Brian McMahan, "Natural Language Processing with PyTorch", O'Reilly Media Inc., 2019.
