# Keras-TensorFlow-GPU-Windows-Installation

Installation instructions for TensorFlow and Keras on Windows.

## 1. Install Miniconda python

Miniconda installers contain the conda package manager and Python. Once Miniconda is installed, you can use the conda command to install any other packages and create environments, etc.

### 1.1: Install Miniconda (Python 3.6 version) <a href="https://conda.io/miniconda.html" target="_blank">Download</a>

<p align="center"><img width=70% src="media\machinesetup\miniconda.png"></p>

### 1.2: Update conda

Run Anaconda Prompt as administrator

<p align="center"><img src="media\machinesetup\prompt.png"></p>

and type the following command(s)

```cmd
conda update conda
conda update --all
```

### 1.3: Install python IDE

Install your favorite python IDE (Visual Studio Code, Python Tools for Visual Studio, PyCharm, Ninja...)


## 2. If you have GPU, install CUDA and cuDNN

### 2.1: Install CUDA Tookit 9.0 <a href="https://developer.nvidia.com/cuda-90-download-archive" target="_blank">Download</a>
Choose your version depending on your Operating System

<p align="center"><img width=70% src="media\machinesetup\CUDA9_0.png"></p>

For more information, refer to <a href="http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html">official documentation.</a>

### 2.2: Download cuDNN <a href="https://developer.nvidia.com/rdp/cudnn-archive" target="_blank">Download</a>
Choose your version depending on your Operating System.
Membership registration is required.

<p align="center"><img width=70% src="media\machinesetup\cuDNN7_05.png"></p>

Put your unzipped folder in C drive as follows:

```cmd
C:\cudnn-9.0-windows10-x64-v7
```

### 2.3: Add cuDNN into Environment PATH <a href="https://kb.wisc.edu/cae/page.php?id=24500" target="_blank">Tutorial</a>

Add the following path in your Environment.
Subjected to changes in your installation path.

```cmd
C:\cudnn-9.0-windows10-x64-v7\cuda\bin
```

Close all prompts.
Open a new command prompt and type the following command

```cmd
echo %PATH%
```

You shall see that the new Environment PATH is there.

## 3. Install TensorFlow

### 3.1: Create an Anaconda environment with Python=3.6

Open Anaconda prompt (**as an administrator**) and type the following command

```cmd
conda create -n tensorflow python=3.6
```

### 3.2: Activate TensorFlow environment

In the command prompt type the following command

```cmd
activate tensorflow
```

### 3.3: Install TensorFlow package

If you have a GPU, install GPU version of TensorFlow by running the following command

```cmd
pip install --ignore-installed --upgrade tensorflow-gpu
```

If you don't have a GPU, install CPU version of TensorFlow by running the following command

```cmd
pip install --ignore-installed --upgrade tensorflow
```

For more information, refer to <a href="https://www.tensorflow.org/install/install_windows">official documentation.</a>

## 4. Install Keras

In the command prompt type the following command

```cmd
pip install keras
```

## 5. Test the installation

Let's try running ```examples/machinesetup/mnist_mlp.py``` in your Anaconda prompt.

Open Anaconda prompt in the ```examples/machinesetup``` folder and type the following commands

```cmd
activate tensorflow
python mnist_mlp.py
```

You should see output similar to this:

```cmd
Using TensorFlow backend.
60000 train samples
10000 test samples
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 512)               401920
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 512)               262656
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_3 (Dense)              (None, 10)                5130
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
_________________________________________________________________
Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 5s 77us/step - loss: 0.2414 - acc: 0.9264 - val_loss: 0.1263 - val_acc: 0.9584
Epoch 2/20
60000/60000 [==============================] - 3s 43us/step - loss: 0.1028 - acc: 0.9690 - val_loss: 0.0846 - val_acc: 0.9746
...
...
...
Epoch 20/20
60000/60000 [==============================] - 3s 44us/step - loss: 0.0163 - acc: 0.9959 - val_loss: 0.1230 - val_acc: 0.9831
Test loss: 0.123033428495213
Test accuracy: 0.9831
```

Congratulations! You have successfully run Keras (with Tensorflow backend) on Windows!