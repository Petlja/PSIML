---
marp: true
$size: 16:9
<!--footer: 'Neural Networks and Backpropagation'-->
---

# Neural Networks and Backpropagation

---

# Agenda

* Neural networks
* Backpropagation
* Best practices
* Workshop

---

# NEURAL NETWORKS

---

# Introduction

* Classification
    * Pattern recognition (image, medical diagnosis, …)
    * Sequence recognition (printed text, handwriting, speech)
* Regression analysis (function approximation)
    * Time series prediction
* Other
    * Novelty detection (fraud detection)
    * Data compression (auto-encoders)
* Great investment in industry and academia

---

# History

* Inspired by human nervous system

![](media/human_nervous_system.png "")

---

# History

* 1960 - Perceptron
    * Limited to linearly separable problems
:::::: container {.two_columns .min_content}
::: container
![](media/perceptron.png "")
:::
::: container
![](media/xor_function.png "")
::::::

  * Insufficient compute power to perform training
* Stagnant until ~1980

---

# History

:::::: container {.two_columns}
::: container
* 1980-1995 – Multilayer perceptron
  * Universal approximator
  * Enough compute power

* 1995 - Stagnant again
  * More complex problems, lacks data and compute
  * Other techniques more popular (SVM)

* 2006 – Deep neural networks
  * Increase number of layers (higher abstraction level)
  * Large datasets GPU compute
* Very popular technique until today
:::
::: container
![](media/nn_simple.png "")
![](media/nn_complex.png "")
::::::
---
# Neural Network types

:::::: container {.two_columns}
::: container
* Feed forward
  * Fully connected

    ![](media/fully_connected.png "")

  * Convolutional
  
    ![](media/convolutional.png "")
:::
::: container
* Recurrent

    ![](media/recurrent.png "")
::::::
---

# FF Neural Networks - Neuron

:::::: container {.two_columns}
::: container
* Neuron – basic building block
  * Input vector

    $$I_{xx}=\int\int_Ry^2f(x,y)\cdot{}dydx$$
  * Weights vector

    $$a^2 + b^2 = c^2$$
  * Bias $b$
:::
::: container

![](media/neuron.png "")

::::::