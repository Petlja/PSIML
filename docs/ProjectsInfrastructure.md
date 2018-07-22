# PSIML 2018

There will be three type of machines available for PSIML 2018 projects:
* Windows machines is the cloud
* Linux machine is the cloud
* Local on premise machines (should be used only if some issue is encountered while using cloud machines).

## Windows cloud machines

Steps to use Linux cloud machines:
* Got to https://app.fra.me/login
* Enter the Windows credentials you received.
* In the upper left corner click _Go To Dashboard_ in the drop down (next to _Microsoft Powered by FRAME_ title).
* In the _Apps_ tab click _POWER ON_ and wait machine to boot. This step can be skipped if machine is already booted.
* After machine is booted click _Open Desktop_ and you will be presented with the desktop in browser.

Miniconda Python distribution is installed at c:\Data\Programs\miniconda. This is the clean installation without any frameworks installed. **One should never use base Miniconda environment but rather create new virtual environment**. To do that follow the steps:
* Activate base miniconda environment
    ```
    C:\Data\Programs\miniconda\3.6\Scripts\activate
    ```
* Create new virtual environment 
    ```
    conda create -p <<path>> python=3.5
    ```
    where _path_ is path to the new virtual environement (for example c:\data\users\newuser\pyenvs\tf-35).
* Activate new virtual environment
    ```
    conda activate <<path>>
    ```
In virtual environment new python packages can be installed without affecting base miniconda install. If something goes wrong virtual environment can be easily removed and recreated.

Once virtual environment is activated one can install required framework packages. For example to install TensorFlow:
```
pip install tensorflow-gpu
```

To facilitate development and debugging following tools are installed:
* Git - for source control
* WinMerge - visual diff viewer
* Visual Studio Code - Lighweight Python IDE

## Linux cloud machines

Steps to use Linux cloud machines:
* Go to https://app.fra.me/login
* Enter the Linux credentials you received.
* In the upper left corner click _Go To Dashboard_ in the drop down (next to _Microsoft Powered by FRAME_ title).
* In the _Apps_ tab click _POWER ON_ and wait machine to boot. This step can be skipped if machine is already booted.
* After machine is booted click _Open Desktop_ and you will be presented with the desktop in browser.

After machine is booted you will be presented with the desktop in browser.

Machine OS is Ubuntu 16.04. Anaconda Python distribution is installed at /home/ubuntu/src/anaconda3 (there is a system-wide Python distribution but that one should not be used). Following ML frameworks pre-installed in anaconda base environment.:
* Apache MXNet 
* MXNet Model Server
* TensorFlow
* TensorBoard
* Caffe
* Caffe2
* PyTorch
* CNTK
* Theano
* Keras
* CUDA 8 and 9

**One should never use base Anaconda environment but rather create new virtual environment**. To do that follow the steps:
1) Activate base anaconda environment
    ```
    source /home/ubuntu/src/anaconda3/bin/activate
    ```
2) Create new virtual environment by copying base environment
    ```
    conda create -p <<path>> --clone /home/ubuntu/src/anaconda3
    ```
    where _path_ is path to the new virtual environement (for example /home/ubuntu/users/newuser/myenv-py35).
3) Activate new virtual environment
    ```
    source activate <<path>>
    ```

In virtual environment new python packages can be installed without affecting base anaconda install. If something goes wrong virtual environment can be easily removed and recreated.

To reactivate virtual environment just repeat steps 1. and 3.

To facilitate development and debugging following tools are installed:
* Git - for source control
* Meld - visual diff viever
* Visual Studio Code - Lighweight Python IDE

## Windows on-prem machines

TBD.
