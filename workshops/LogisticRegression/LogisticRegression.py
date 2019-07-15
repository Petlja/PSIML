# <markdowncell>
# # Numpy

# <markdowncell>
# NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.

# <codecell>
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 

# <markdowncell>
# ## Arrays

# <codecell>
# Define array
a = np.array([1,2,3])

# Some basic properties
print("Array a: ", a)
print("\nShape of array a: ", a.shape)
print("\nData type of array a: ", a.dtype)

# <codecell>
# Define matrix
b = np.array([[1, 2, 3], [4, 5, 6]])

# Some basic properties
print("Matrix b: \n", b)
print("\nShape of matrix b: ", b.shape)
print("\nData type of matrix b: ", b.dtype)


# <codecell>
# Multidim arrays - tensor
c = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

# Some basic properties
print("Tensor c: \n", c)
print("\nShape of tensor c: ", c.shape)
print("\nData type of tensor c: ", c.dtype)

# <markdowncell>
# ## Initialization functions

# <codecell>
# All zeros
print("All zeros: \n", np.zeros((2,2)))

# All ones
print("\nAll ones: \n", np.ones((2,2)))

# All same value
print("\nAll same value: \n", np.full((2,2), 2))

# All random
# Setting a random seed is important for reproducibility of the code.
# It is good practice to use it in ML before moving to actual training as it makes debuging a lot easier.
np.random.seed(5)
print("\nAll random: \n", np.random.random((2,2)))

# Identity matrix
print("\nIdentity matrix: \n", np.eye(3))

# <markdowncell>
# ## Array indexing
#
# Indexing starts from 0. It is possible to use negative indexes (for example -1 for last element of array)

# <codecell>
print("Array a: ", a)
print("First element of a: ", a[0])
print("Last element of a: ", a[2])
print("Last element of a: ", a[-1])

# <markdowncell>
# Indexing in matrix and tensor is the same and we can index any column, row etc.


# <codecell>
print("Tensor c: \n", c)
print("\nValue of c[0]: \n", c[0])
print("\nValue of c[-2]: \n", c[-2])
print("\nValue of c[0][1]: ", c[0][1])
print("Value of c[0][0][0]: ", c[0][0][0])
print("Value of c[0, 0, 0]: ", c[0, 0, 0])
print("\nValue of c[0, :, 0:2]: \n", c[0, :, 0:2])

# <markdowncell>
# ## Basic operations

# <codecell>
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

print("Matrix x: \n", x)
print("\nMatrix y: \n", y)


# <codecell>
print("Addition:\n", x + y)
print("Substruction:\n", y - x)
print("Elementwise multiplication:\n", x * y)
print("Multiplication:\n", np.matmul(x, y))
print("Divion:\n", x / y)
print("Square root:\n", np.sqrt(x))
print("Exp:\n", np.exp(x))
print("Dot product:\n", np.dot(x[1], y[0]))
print("Transpose:\n", x.T)
print("Inverse:\n", np.linalg.inv(x))

# <markdowncell>
# ## Broadcasting
#
# Broadcasting is one of the most important numpy features. The term broadcasting describes how numpy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is "broadcast" across the larger array so that they have compatible shapes. Broadcasting provides a means of vectorizing array operations so that looping occurs in C instead of Python. It does this without making needless copies of data and usually leads to efficient algorithm implementations. 

# <codecell>
a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 2.0, 2.0])
print("a * b, a as vector, b as vector:", a * b)

b = np.array([2])
print("a * b, a as vector, b as scalar:", a * b)


# <codecell>
a = np.array([[1,2,3], [4,5,6]])
b = np.array([2,4,6])

print("a + b, a as matrix, b as vector:\n", a + b)
print("a * b, a as matrix, b as vector:\n", a * b)
print("Dot product of a and b:\n", np.dot(a, b))

# <markdowncell>
# ## Important ML functions:
# ### Sigmoid function:
#
# \begin{equation*}
# S(x) = \frac{1}{1 + e^{-x}}
# \end{equation*}
#
# You can find more at *https://en.wikipedia.org/wiki/Sigmoid_function*

# <codecell>
def sigmoid(x):
    # [TODO] Implement sigmoid function
    return 0

# <codecell>
print("Sigmoid of \"0\":", sigmoid(0))
print("Expected value: 0.5")
testArray = np.array([1,5])
print("Sigmoid of [1,5]:", sigmoid(testArray))
print("Expected value: [0.73105858 0.99330715]")

# <markdowncell>
# ### Ploting Sigmoid


# <codecell>

x = np.arange(-10., 10., 0.2)
y = sigmoid(x)
plt.plot(x,y)
plt.show()




# <markdowncell>
# # Linear Regression

# <markdowncell>
# ## Normal equations
#
# Standard variant
# \begin{equation}
# w = (X^TX)^{-1}X^Ty
# \end{equation}
#
# Regularized variant
# \begin{equation}
# w = (X^TX+\lambda I)^{-1}X^Ty
# \end{equation}

# <codecell>
# Fixing random seed
np.random.seed(5)

# Generating synthetic data from exponential function with some noise
sampleSize = 20
x = sorted(np.random.uniform(0, 4, sampleSize))
y = np.exp(x) + np.random.normal(0, 0.01, sampleSize)
plt.plot(x, y, 'bo')
plt.show()

# Save the original vector x because we will be transforming our features
xo = x;

# Adding column of ones for implicit treatment of bias term
x = np.concatenate((np.ones((sampleSize,1)), np.reshape(xo, (sampleSize, 1))), 1)

# <codecell>
# [TODO] Estimate w via normal equations


# [TODO] Compute regression values z using w



# <codecell>
# Plot predictions against x (using original x)
plt.plot(xo, y, 'bo');
plt.plot(xo, z)
plt.show()

# <codecell>
# Generating powers of x up to some degree
degree = 12
for i in range(1, degree):
    t = np.reshape(x[:, -1] * xo, (sampleSize, 1))
    x = np.concatenate((x, t), 1)
    
# <codecell>
# Play with regularization parameter to tune ridge regression 
# Start from 10^-6 and increase exponentially
lmbd = 0

# [TODO] Estimate w using regularized normal equations


# [TODO] Compute regression values z using w


# <codecell>
# Plot predictions against x
plt.plot(xo, y, 'bo');
plt.plot(xo, z)
plt.show()


# <markdowncell>
# # Logistic Regression

# <markdowncell>
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

# <markdowncell>
# ## Reading the dataset

# <codecell>
import numpy as np
import csv

dataPath = r'Data\train.csv'

def readCSVasNumpy(dataPath):
    with open(dataPath,'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"')
        data = [data for data in data_iter]
    data_array = np.asarray(data, dtype = None)
    return data_array

data = readCSVasNumpy(dataPath)

# <markdowncell>
# ## Exploring the dataset 

# <codecell>
print(data)


# <codecell>
labels = (data[1:,1]).astype(int)

print(labels)

# <markdowncell>
# ## Manual feature selection

# <codecell>
print(data[0:2,:])


# <codecell>
# Select some features
important_fields = [2, 4, 5, 6, 7, 9]
features = data[1:, important_fields]
print(features)

# <markdowncell>
# ## Trivial dummy coding

# <codecell>
features[:,1] = (features[:,1]=="male").astype(float)
print(features)

# <markdowncell>
# ## Train/Test split

# <codecell>
trainIndexes = np.sort(np.random.choice(features.shape[0], int(features.shape[0]*0.7), replace=False))

trainFeatures = features[trainIndexes]
testFeatures = np.delete(features, trainIndexes, axis=0)

trainLabels = labels[trainIndexes]
testLabels = np.delete(labels, trainIndexes, axis=0)

# <markdowncell>
# ## Missing data imputation

# <codecell>
# Identify columns with missing data

print(np.sum(trainFeatures == "", 0))
print(np.sum(testFeatures == "", 0))
print(trainFeatures.shape)
print(testFeatures.shape)

# <codecell>
# Compute average of existing values on the TRAINING SET and 
# use it to substitute missing values in both sets

# Column with missing values
col = 2

agePresentMask = np.where(trainFeatures[:,col] != "")
averageAge = np.mean(trainFeatures[agePresentMask,col].astype(float))
trainFeatures[np.where(trainFeatures[:,col] == ""),col] = str(averageAge)

testFeatures[np.where(testFeatures[:,col] == ""),col] = str(averageAge)


# <markdowncell>
# ## Feature normalization


# <codecell>
# Convert everything to floats so we can transform features

trainFeatures = trainFeatures.astype(float)
testFeatures = testFeatures.astype(float)
print(trainFeatures)
print(testFeatures)


# <codecell>
# Scale features to interval [0,1]

maxFeatures = np.max(trainFeatures, axis=0)
minFeatures = np.min(trainFeatures, axis=0)

trainFeatures = (trainFeatures - minFeatures) / (maxFeatures - minFeatures)
testFeatures = (testFeatures - minFeatures) / (maxFeatures - minFeatures)

print(trainFeatures)
print(testFeatures)


# <codecell>
# Adding column of ones for implicit treatment of bias term

trainFeatures = np.concatenate((np.ones((trainFeatures.shape[0],1)), trainFeatures), 1)
testFeatures = np.concatenate((np.ones((testFeatures.shape[0],1)), testFeatures), 1)
print(trainFeatures)
print(testFeatures)

# <markdowncell>
# ## Logistic Model
#
# \begin{equation*}
# f_w(x) = \frac {1}{1+e^{-\sum_{i=0}^n{w_i x_i}}}
# \end{equation*}


# <codecell>

class LRmodel:
    def __init__(self, weights):
        self.w = weights
        
    def __init__(self, numFeatures):
        self.w = np.ones(numFeatures)
            
    def predict(self, features):   
		# [TODO] Implement prediction based on the formulas above
        return 0
    
    def getModelParams(self):
        return self.w
    
    def setModelParams(self, w):
        self.w = w


model = LRmodel(trainFeatures.shape[1])

print("Model weights: ", model.w)
print("Expected values: [1. 1. 1. 1. 1. 1. 1.]")

print("Feature vector shape: ", trainFeatures.shape)
print("Expected values: (623, 7)")

print("First 3 model evaluations: ", model.predict(trainFeatures)[0:3])
print("Expected values: 0.96795449 0.84042498 0.96918524")



# <markdowncell>
# ## Trainer
# 
# ### Loss function:

# \begin{equation*}
# E(w) = \frac {1} {N} \sum_{i=1}^N{L(f_w(x_i),y_i)} 
# \end{equation*}
# \begin{equation*}
# E(w) = \frac {1} {N} \sum_{i=1}^N{[-y_i\log(f_w(x_i)) - (1-y_i)\log(1 -f_w(x_i))]} 
# \end{equation*}
# 
# ### Gradient descent:
# 
# \begin{equation*}
# w_0 = w_0 - \mu \frac{1}{N}\sum_{i=1}^N {(f_w(x_i) - y_i)}
# \end{equation*}
# 
# \begin{equation*}
# w_j = w_j - \mu \frac{1}{N}\sum_{i=1}^N {(f_w(x_i) - y_i) x_{ij}}
# \end{equation*}


# <codecell>

class Trainer:
    def __init__(self, model):
        self.model = model
    
    def calculateLoss(self, features, labels):
        # [TODO] Implement loss function based on the formulas above
        return 0
    
    def calculateGradients(self, features, labels):
        # [TODO] Implement gradient function based on the formulas above
        return 0
    
    def updateModel(self, gradient, learningRate):
        # [TODO] Implement model update based on the gradients
    
    def train(self, features, labels, learningRate, iters, lossValues):
        for i in iters:
			# [TODO] Implement one itteration of training

			
            loss = self.calculateLoss(features, labels)    
            lossValues.append(loss)


# <markdowncell>
# ## Training


# <codecell>

model = LRmodel(trainFeatures.shape[1])
trainer = Trainer(model)

print("Starting loss training: ", trainer.calculateLoss(trainFeatures, trainLabels))

learningRate = 30
lossValues = []
iters = np.arange(1, 2000, 1)

trainer.train(trainFeatures, trainLabels, learningRate, iters, lossValues)

lossValues = np.array(lossValues)
print("End loss training: ", lossValues[-1])

plt.figure(1, figsize=(20, 8))
plt.plot(iters, lossValues)
plt.show()

# <markdowncell>
# ## Evaluation

# <codecell>

class Evaluator:
    def __init__(self, model):
        self.model = model
        
    def evaluate(self, features):
        predictions = self.model.predict(features)
        return predictions
    
    def calculateAPR(self, features, labels, threshold):
        predictions = self.evaluate(features)
        
        numExamples = predictions.shape[0]
        binaryPredictions = (predictions > threshold).astype(int)
        
        positivePredictions = np.where(binaryPredictions == 1)
        negativePredictions = np.where(binaryPredictions == 0)
        
        # [TODO] Implement calculation of TP, FP, TN, FN, Precision, Recall and Accuracy
        
        # TP - Count of examples that were correctlly predicted as positive examples
        
        # FP - Count of examples that were incorectlly predicted as positive examples
        
        # FN - Count of examples that were incorectlly predicted as negative examples
        
        # TN - Count of examples that were correctlly predicted as negative examples
        
        
        return Precision, Recall, Accuracy
    
    def plotAPR(self, resultsTest, resultsTrain, ranges):
        plt.figure(1, figsize=(20, 15))
        plt.subplot(211)
        plt.plot(ranges, np.matrix(resultsTrain)[:,0], ranges, np.matrix(resultsTrain)[:,1], ranges, np.matrix(resultsTrain)[:,2])
        plt.subplot(212)
        plt.plot(ranges, np.matrix(resultsTest)[:,0], ranges, np.matrix(resultsTest)[:,1], ranges, np.matrix(resultsTest)[:,2])
        plt.show()


# <markdowncell>
# ## Evaluation of the model


# <codecell>

evaluator = Evaluator(model)
t = np.arange(0., 1., 0.001)
resultsTest = []
resultsTrain = []
for i in t:
    resultsTest.append(evaluator.calculateAPR(testFeatures, testLabels, i))
    resultsTrain.append(evaluator.calculateAPR(trainFeatures, trainLabels, i))

evaluator.plotAPR(resultsTest, resultsTrain, t)

print("Model w: ", model.w)
print("Accuracy: ", np.sum((model.predict(testFeatures)>0.5) == (testLabels==1))/testFeatures.shape[0])
