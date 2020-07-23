# <markdowncell>
# # Numpy

# <markdowncell>
# NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.

# <codecell>
import numpy as np

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
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)

# Some basic properties
print("Matrix b: \n", b)
print("\nShape of matrix b: ", b.shape)
print("\nData type of matrix b: ", b.dtype)

# <codecell>
# Multidim arrays - tensor
c = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]], dtype=np.float64)

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
print("\nAll ones: \n", np.ones((2,3,4)))

# All same value
print("\nAll same value: \n", np.full((2,2), 2))

# Identity matrix
print("\nIdentity matrix: \n", np.eye(3))

# Linspace
print("\nUniformly spaced elements: \n", np.linspace(start=0, stop=1, num=10, endpoint=False))

# Arange
print("\nNumpy array with range of numbers: \n", np.arange(start=0, stop=10, step=2, dtype=np.float64))

# <markdowncell>
# Numpy offers a variety of ways to generate "random" numbers.

# <codecell>
# All random
# Setting a random seed is important for reproducibility of the code.
# It is good practice to use it in ML before moving to actual training as it makes debuging a lot easier.
np.random.seed(5)
print("\nRandom uniform: \n", np.random.random((2,2)))

# If high=None, then [0, low) is returned. By default, single integer is returned if size=None
print("\nRandom integer: \n", np.random.randint(low=2, high=9, size=(3, 3)))
print("\nRandom choice: \n", np.random.choice([1, 2, 3, 4, 5, 6], size=(4)))
print("\nSample from Binomial distribution: \n", np.random.binomial(n=10,p=0.5))

# <markdowncell>
# ## Array indexing
# Indexing starts from 0. It is possible to use negative indexes (for example -1 for last element of array)

# <codecell>
print("Array a: ", a)
print("First element of a: ", a[0])
print("Last element of a: ", a[2])
print("Last element of a: ", a[-1])

# <markdowncell>
# Indexing in matrix and tensor is the same and we can index any column, row etc.

# <codecell>
print("Matrix b: \n", b)
print("\nValue of b[0]: \n", b[0])
print("\nValue of b[-2]: \n", b[-2])
print("\nValue of b[0][1]: ", b[0][1])
print("Value of b[0, 1]: ", b[0, 1])
print("\nValue of b[0, :]: \n", b[0, :])
print("\nValue of b[0:2, 1:]: \n", b[0:2, 1:]) # General form is start:end

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
print("Subtraction:\n", y - x)
print("Elementwise multiplication:\n", x * y)
print("Multiplication:\n", np.matmul(x, y))
print("Multiplication x@y:\n", x@y)
print("Divison:\n", x / y)
print("x>2:\n", x>2)
print("Square root:\n", np.sqrt(x))
print("Exp:\n", np.exp(x))
print("Dot product:\n", np.dot(x[1], y[0]))
print("Transpose:\n", x.T)
print("Inverse:\n", np.linalg.inv(x))
print("Determinant:\n", np.linalg.det(y))

# <codecell>
w = [0, 3, -2, 9, 5]
print("w:\n", w)
print("Argmax w:\n", np.argmax(w))
print("Max w:\n", np.max(w))

# <markdowncell>
# Boolean indexing

# <codecell>
arr = np.array([0, -3, 4, -1, 2, 8, 5])
print("arr:\n", arr)
print("Boolean indexing:\n", arr[[False, False, True, False, False, True, True]])
print("arr>2:\n", arr>2)
print("arr[arr>2]:\n", arr[arr>2])
print("Remainder of division by 2:\n", np.remainder(c, 2)==0)
print("Even numbers:\n", c[np.remainder(c, 2)==0])

# <markdowncell>
# ## Practice problem: Solve system of equations
# Solve the system of 5 equations involving 5 unknowns:
# \begin{equation*}
# x_1-7x_5 = 2
# \end{equation*}
# \begin{equation*}
# x_2+9x_5=1
# \end{equation*}
# \begin{equation*}
# x_3+x_5=3
# \end{equation*}
# \begin{equation*}
# x_4-2x_5=5
# \end{equation*}
# \begin{equation*}
# 3x_1+2x_2-x_3+4x_5=9
# \end{equation*}

# <codecell>
x = np.zeros(5) # This is solution vector

# [TODO] Solve the system and fill the vector x

print("Solution:\n")
print("x1=", x[0])
print("x2=", x[1])
print("x3=", x[2])
print("x4=", x[3])
print("x5=", x[4])

# <markdowncell>
# ## Broadcasting
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

print("Matrix a:\n", a)
print("Vector b:", b)
print("a + b, a as matrix, b as vector:\n", a + b)
print("a * b, a as matrix, b as vector:\n", a * b)
print("Dot product of a and b:\n", np.dot(a, b))

# <markdowncell>
# ## Practice problem: Implement sigmoid function:
# ### Sigmoid function:

# \begin{equation*}
# S(x) = \frac{1}{1 + e^{-x}}
# \end{equation*}

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