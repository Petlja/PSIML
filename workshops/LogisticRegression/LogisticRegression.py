#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# ### Data
# Problem and data taken from *https://www.kaggle.com/c/titanic*
# 
# ### Goal 
# 
# Based on the provided information about person predict if person survived Titanic crash or not.
# 
# ### Feature explanation
# 
# | Variable | Definition | Key |
# | ------------- | ------------- | ------------- |
# | survival | Survival | 0 = No, 1 = Yes |
# | pclass | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd |
# | sex | Sex | |
# | Age | Age in years | |
# | sibsp | # of siblings / spouses aboard the Titanic | |
# | parch | # of parents / children aboard the Titanic | |
# | ticket | Ticket number | |
# | fare | Passenger fare | |
# | cabin | Cabin number | |
# | embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |
# 
# ### Variable Notes
# 
# **pclass**: A proxy for socio-economic status (SES)  
# 1st = Upper  
# 2nd = Middle  
# 3rd = Lower  
# 
# 
# **age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5  
# 
# **sibsp**: The dataset defines family relations in this way...  
# Sibling = brother, sister, stepbrother, stepsister  
# Spouse = husband, wife (mistresses and fiancés were ignored)  
# 
# **parch**: The dataset defines family relations in this way...  
# Parent = mother, father  
# Child = daughter, son, stepdaughter, stepson  
# 
# Some children travelled only with a nanny, therefore parch=0 for them.  

# ## Reading the dataset

# In[ ]:


import os
import pandas as pd

data_path = os.path.join('Data', 'train.csv')
data = pd.read_csv(data_path)


# ## Exploring the dataset 

# In[ ]:


data.sample(n=10)


# In[ ]:


data.describe()


# ## Feature engineering
# Unlike the vast majority of deep learning techniques, data science problems usually require a lot of feature manipulation. It's always a good thing to have an expert in the field take a look at the data and provide input about what makes sense, etc. For example, in our case by domain knowledge, we conclude that a person's name should not be of any value. The next thing to do is to analyze raw attributes and come up with good features candidates. Some of the attributes are going to end up straight as features in our models, some will be removed, some will get replaced by brand new features. An example of the last one is creating dummy coding out of categorical attributes.
# 
# ### Guidelines for creating features
# 1. Consider removing features which domain experts characterized as unimportant
# 1. Perform missing values imputation
# 1. Analyze correlation of each feature with the target variable (and with each other)
# 1. Remove highly correlated features
# 1. Encode categorical variables in a sensible way (e.g. using dummy coding)
# 
# ### Other things to try
# 1. Perform clustering of the data, and include cluster_id as a feature
# 1. Perform dimmensionality reduction technique (e.g. PCA)

# ## Remove unnecessary attributes

# In[ ]:


# Before we apply any transformation on the data, it's a good idea to copy the data on a safe
# place in order to have a "raw" copy just as it was loaded from a file.
df = data.copy()
df.shape


# In[ ]:


df = df.drop(columns=['PassengerId', 'Name', 'Ticket'])


# ## Missing values imputation
# Pandas library provides us with a useful function: [DataFrame.isna()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isna.html) which will return an array, every value in a dataframe will be replaced by a boolean value indicating whether this value is NULL or not. We can use this as a mask to select only rows containing NULL values for certain attributes.

# In[ ]:


# Example usage of .isna() function
df.isna()


# In[ ]:


df['Embarked'].isna()


# In[ ]:


embarked_null_mask = df['Embarked'].isna()
df[embarked_null_mask]


# In[ ]:


df_missing = pd.concat([ df.isna().sum(),  100*df.isna().sum()/891], axis=1)
df_missing.columns = ['# missing', '% missing']
df_missing


# The simplest ways to impute missing values are:
# * Remove attributes that contain missing values
# * Remove instances that contain missing values
# * Replace missing values with mean value of that feature (we will use Panda's [DataFrame.fillna()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html) function)

# In[ ]:


# Fix missing values in attribute "Age" by replacing them with mean value of this attribute
age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(age_mean)

# Sanity check
df_missing = pd.concat([ df.isna().sum(),  100*df.isna().sum()/891], axis=1)
df_missing.columns = ['# missing', '% missing']
df_missing


# In[ ]:


# Fix missing values in attribute "Cabin" by removing the attribute
df = df.drop(columns=['Cabin'])

# Sanity check
df_missing = pd.concat([ df.isna().sum(),  100*df.isna().sum()/891], axis=1)
df_missing.columns = ['# missing', '% missing']
df_missing


# In[ ]:


# Fix missing values in attribute "Embarked" by removing instances which contain this missing value
df = df[~df['Embarked'].isna()]

# Sanity check
df_missing = pd.concat([ df.isna().sum(),  100*df.isna().sum()/891], axis=1)
df_missing.columns = ['# missing', '% missing']
df_missing


# In[ ]:


pd.get_dummies(df)


# In[ ]:


def get_correct_dummies(df, categorical_attributes):
    df_copy = df.copy()
    for attr in categorical_attributes:
        dummy = pd.get_dummies(df_copy[attr], columns=[attr], prefix=attr)
        without_last = dummy.drop(columns=[dummy.columns.values[-1]])
        df_copy = df_copy.drop(columns=[attr])
        df_copy = pd.concat([df_copy, without_last], axis=1)
    return df_copy
    
df_encoded = get_correct_dummies(df, ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'])
df_encoded


# ## Analyze correlation

# In[ ]:


import matplotlib.pyplot as plt

f = plt.figure(figsize=(19, 9))
plt.matshow(df_encoded.corr(), fignum=f.number)
plt.xticks(range(df_encoded.shape[1]), df_encoded.columns, fontsize=14, rotation=90)
plt.yticks(range(df_encoded.shape[1]), df_encoded.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)


# Let's now separate features from label and remove redundant features (ones that are highly correlated with other features). For example, Fare has high correlation with feature Pclass_1, so we will remove one of them. We selected Pclass_1 but since it's only a part of encoded attribute Pclass, we will have to remove Pclass_2 as well. High correlation between parts of encoded variables could be left as is. They are somewhat expected because of the way we built those encodings. We also observe that Parch is correlated to SibSp attribute, but if we removed one of them we'd end up with a small number of features so for the demonstration purposes let them be included in the final dataset.

# In[ ]:


df_features = df_encoded.drop(columns=['Survived', 'Pclass_1', 'Pclass_2'])
print(f'Number of features in a dataset: {df_features.shape[1]}')

df_labels = df_encoded[['Survived']]


# In[ ]:


# Conversion from Pandas dataframes to Numpy nd-arrays
features = df_features.to_numpy()
labels = df_labels.to_numpy().ravel()

print(features.shape)
print(labels.shape)


# ## Train/Test split

# In[ ]:


import numpy as np

# We will fix pseudo-random number generator so that we all get the same results
np.random.seed(0)

train_indices = np.sort(np.random.choice(features.shape[0], int(features.shape[0]*0.7), replace=False))

train_features = features[train_indices]
test_features = np.delete(features, train_indices, axis=0)

train_labels = labels[train_indices]
test_labels = np.delete(labels, train_indices, axis=0)


# ## Feature normalization

# In[ ]:


# Convert everything to floats so we can transform features

train_features = train_features.astype(float)
test_features = test_features.astype(float)


# In[ ]:


# Standardize features to have mean=0, and std=1

mean_train = np.mean(train_features, axis=0)
std_train = np.std(train_features, axis=0)

train_features = (train_features - mean_train) / std_train
test_features = (test_features - mean_train) / std_train

print(f'Features mean: {train_features.mean(axis=0).round()}')
print(f'Features std: {train_features.std(axis=0).round()}')


# In[ ]:


# Adding column of ones for implicit treatment of bias term

train_features = np.concatenate((np.ones((train_features.shape[0],1)), train_features), 1)
test_features = np.concatenate((np.ones((test_features.shape[0],1)), test_features), 1)


# ## Important ML functions:
# ### Sigmoid function:
# 
# \begin{equation*}
# S(x) = \frac{1}{1 + e^{-x}}
# \end{equation*}
# 
# You can find more at *https://en.wikipedia.org/wiki/Sigmoid_function*

# In[ ]:


def sigmoid(x):
    return np.zeros(x.shape)  # [TODO] Implement sigmoid function


# In[ ]:


print("Sigmoid of \"0\":", sigmoid(np.array([0])))
print("Expected value: 0.5")
testArray = np.array([1,5])
print("Sigmoid of [1,5]:", sigmoid(testArray))
print("Expected value: [0.73105858 0.99330715]")


# In[ ]:


x = np.arange(-10., 10., 0.2)
y = sigmoid(x)
plt.plot(x,y)
plt.show()


# ### Logistic Model
# 
# \begin{equation*}
# f_w(x) = \frac {1}{1+e^{-\sum_{i=0}^n{w_i x_i}}} = \frac {1}{1+e^{-\mathbb{x} \mathbb{w}}}\\
# \end{equation*}
# 
# ### Loss function
# \begin{equation*}
# L(w) = \frac {1} {N} \sum_{i=1}^N{L(f_w(x_i),y_i)} = \frac {1} {N} \sum_{i=1}^N{[-y_i\log(f_w(x_i)) - (1-y_i)\log(1 -f_w(x_i))]} 
# \end{equation*}
# 
# ### Gradients
# 
# \begin{equation*}
# w_0 = w_0 - \mu \frac{1}{N}\sum_{i=1}^N {(f_w(x_i) - y_i)}
# \end{equation*}
# 
# \begin{equation*}
# w_j = w_j - \mu \frac{1}{N}\sum_{i=1}^N {(f_w(x_i) - y_i) x_{ij}}
# \end{equation*}

# In[ ]:


class LogisticRegressionPSIML:
    def __init__(self, num_features):
        np.random.seed(0)
        self.__weights = self.__weights = np.random.rand(num_features)

    def predict(self, features):
        return np.zeros(features.shape[0])  # [TODO] Implement prediction based on the formulas above
        
    def loss(self, predictions, labels):
        return 0  # [TODO] Implement loss calculation based on the formulas above
    
    def fit(self, features, labels, lr, max_iter=1000, eps=10e-5):
        loss_history = []

        predictions = self.predict(features)   
        loss = self.loss(predictions, labels)
        loss_history.append(loss)
        
        for i in range(max_iter):
            if len(loss_history) > 2 and np.isclose(loss_history[-2], loss_history[-1], atol=eps):
                break

            # [TODO] Implement gradient descent step, based on the formulas above
            
            # [CODE ENDS HERE]
            
            predictions = self.predict(features)
            loss = self.loss(predictions, labels)
            loss_history.append(loss)
        return np.array(loss_history)


# In[ ]:


model = LogisticRegressionPSIML(num_features=train_features.shape[1])
predictions = model.predict(train_features)

print('CHECK: Predictions on the first three instances:')
print(f'Calculated predictions before training:\t{predictions[:3]}')
print('Expected predictions before training:\t[0.33532259 0.90135813 0.52439258]')
print('\n')

loss = model.loss(predictions, train_labels)
print('CHECK: Loss before training:')
print(f'Calculated loss before training:\t{loss}')
print('Expected loss before training:\t\t0.484539684559608')
print('\n')

loss_history = model.fit(train_features, train_labels, lr=0.015)
print(f'Training finished after {loss_history.size} epochs')
predictions = model.predict(train_features)


# In[ ]:


f = plt.figure(figsize=(16,9))
plt.plot(loss_history)


# In[ ]:


np.random.seed(1)
rand_ind = np.random.choice(train_labels.shape[0], 5, replace=False)
pd.DataFrame(data=np.stack([train_labels[rand_ind], predictions[rand_ind].round()], axis=1), columns=['Real Labels', 'Predictions'], dtype=int)


# ## Evaluation
# For many different metrics, a usefull thing to compute is a confusion matrix. This is the matrix of the following form:
# 
# \begin{equation*}
# \begin{array} {|r|r|}\hline TP & FP \\ \hline FN & TN \\ \hline  \end{array}
# \end{equation*}
# 
# Where the entries are as following:
# * True Positive (TP) - Number of correctly predicted positive examples (where Survive = 1)
# * True Negative (TN) - Number of correctly predicted negative examples
# * False Positive (FP) - Number of predictions where the model falsly predicted positive value (the model predicted Survive = 1 where it should be Survive = 0)
# * False Negative (FN) - Number of predictions where the model falsly predicted negative value
# 
# After calculating the confusion matrix, interesting metrics to compute are:
# \begin{equation*}
# Accuracy = \frac {TP+TN}{TP+TN+FP+FN}\\
# Precision = \frac {TP} {TP + FP}\\
# Recall = \frac {TP} {TP + FN}
# \end{equation*}
# 
# NOTICE: Model will return probabilities! In order for these metrics to be calculated, these probabilities must be thresholded!

# In[ ]:


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2))
    cm[0, 0] = None  # [TODO] Calculate TP
    cm[0, 1] = None  # [TODO] Calculate FP
    cm[1, 0] = None  # [TODO] Calculate FN
    cm[1, 1] = None  # [TODO] Calculate TN
    
    return cm

def accuracy(y_true, y_pred):
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred).ravel()
    
    return 0  # [TODO] Calculate accuracy

def precision(y_true, y_pred):
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred).ravel()
    
    return 0  # [TODO] Calculate precision

def recall(y_true, y_pred):
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred).ravel()
    
    return 0  # [TODO] Calculate recall


# In[ ]:


test_predictions = model.predict(test_features).round()
print(f'Accuracy: {accuracy(test_labels, test_predictions)}')
print(f'Precision: {precision(test_labels, test_predictions)}')
print(f'Recall: {recall(test_labels, test_predictions)}')


# ## Sci-Kit Learn

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

sk_features = df_features.to_numpy()
sk_labels = df_labels.to_numpy().ravel()

sk_train_features, sk_test_features, sk_train_labels, sk_test_labels = train_test_split(sk_features, sk_labels, stratify=sk_labels, test_size=0.3)

scaler = StandardScaler()
scaler.fit(sk_train_features)
sk_train_features = scaler.transform(sk_train_features)
sk_test_features = scaler.transform(sk_test_features)

model = LogisticRegression(max_iter=1000, penalty='none')
model.fit(sk_train_features, sk_train_labels)
sk_test_predictions = model.predict(sk_test_features).round()

print(f'Accuracy: {accuracy_score(sk_test_labels, sk_test_predictions)}')
print(f'Precision: {precision_score(sk_test_labels, sk_test_predictions)}')
print(f'Recall: {recall_score(sk_test_labels, sk_test_predictions)}')


# In[ ]:




