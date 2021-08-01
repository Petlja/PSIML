### Machine Learning Summer School 2021
# Reinforcement Learning Workshop

## About

This repository contains the code used in RL workshop 2021. The code contains **three assignments** and is split into two main packages:

* *deep_rl* and *rl* folders contain the solutions to all three assignments that can be run locally, in VSCode, PyCharm, or any other editor of choice. These packages do not require the installation of Jupyter notebooks.
* *jupyter* contains the same code as *deep_rl* and *rl*, only formatted for Jupyter notebooks:
    * *jupyter/q_learning.ipynb*: Is **Assignment 1** of this year's RL workshop.
    * *jupyter/q_learning_solutions.ipynb*: Contains the solutions to **Assignment 1**.
    * *jupyter/deepq_learning.ipynb*: Is **Assignment 2** of this year's RL workshop.
    * *jupyter/deepq_learning_solutions.ipynb*: Contains the solutions to **Assignment 2**.
    * *jupyter/ddpg_learning.ipynb*: Is **Assignment 3** of this year's RL workshop.
    * *jupyter/ddpg_learning_solutions.ipynb*: Contains the solutions to **Assignment 3**.
    
**If you are reading this prior to the time the workshop is scheduled to be held (03.08.2021), you will find that packages *deep_rl*, *rl*, as well as *jupyter/q_learning_solutions.ipynb*, *jupyter/deepq_learning_solutions.ipynb*, and *jupyter/ddpg_learning_solutions.ipynb* are not available. They will be published once the workshop is over.**

## Local installation instructions

This year's workshop will be held using virtual instances (VM) which will be assigned to each participant, where all dependencies are already installed and properly set up. If you wish, however, to attend the workshop and run the code locally, or to experiment with the code after the workshop is over, you will need to install all dependencies. 

**There are two ways the required resources for this workshop can be installed**:
1. If you wish to install resources for all workshops at once, so there is no potential conflict between workshops you can follow the instructions [here](../ReadMe.md) under *Prerequisites* and *Setting up the environment*.
2. If you want to use just this workshop, and are comfortable managing python dependencies in your own way, you can use the installation instructions provided below.

The installation instructions are provided below. **They assume that you already have Python 3.x and git set up on your machine.** If you do not have Python 3.x, you can check out [this guide](https://realpython.com/installing-python/) to help you with the process of installation. For git refer to [this guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), or simply use the GitHub visual interface to download the code from this repository.

1. Create an empty directory on your local machine where you wish to download the resources. Navigate to the folder and open a terminal at its location (for Linux / Unix systems), or right-click the folder and open a git shell (for Windows). Paste the following command: `git clone https://github.com/Petlja/PSIML.git`. Of course, if you wish to download the code in any other way, you can do so. 
2. RL workshop code uses Python 3.x and has six main dependencies.
    * NumPy: used for numerical computations.
    * PyTorch: used for numerical computations.
    * OpenAI Gym: used for environments in deep learning exercises. 
    * Matplotlib: used for result plotting in deep learning exercises.
    * Jupyter: used as an executing environment for assignments.
    * jdc: used for presenting jupyter notebooks in a cleaner way. (must be installed after jupyter!)
    
    You can install these dependencies in any way you are comfortable with. If you are using the pip package manager you can run the following code in your Linux/Unix terminal or Windows command prompt:
    * `pip3 install numpy`
    * `pip3 install torch`
    * `pip3 install gym`
    * `pip3 install matplotlib`
    * `pip3 install jupyter`
    * `pip3 install jdc`
      
    If you are using conda environment, you can:
    * `conda install numpy`
    * `conda install pytorch`
    * `conda install gym`
    * `conda install matplotlib`
    * `conda install jupyter`
    * `conda install jdc`
    
    **In any case, please make sure that you are installing these libraries for the python interpreter you will be using to run the workshop code!** If you are using conda environment this should be guaranteed, while with pip you can type `which pip3` in the Linux/Unix terminal or Windows command prompt to check where it located. If it's located in the directory of python you will be using to run the code you are all good.

## Running the code and verifying the installation

To check whether the installation was successful:

* Navigate to the *jupyter* directory  of ReinforcementLearning2021 workshop, located in workshops, and type `jupyter notebook` in the Linux/Unix terminal or Windows command prompt. **Note: It is important to be in the *ReinforcementLearning2021/jupyter* directory when running this command so that everything renders properly in the notebooks!**
* Open either *deepq_learning.ipynb* or *ddpg_learning.ipynb* notebook. Navigate to the **second** code cell (the first one displays some video, the second one imports all libraries that we wish to verify are installed properly). Run the second code cell. If it executes without issues, everything is OK :)
* Steps to try if everything is not OK:
    * Verify that jupyter python interpreter is the interpreter you've installed dependencies from previous steps for. 
    * You can try to install all requirements from *requirements.txt* file, by running `pip3 install -r requirements.txt`, or `conda install requirements.txt`. These additional requirements should have been already installed as they are dependencies of the base 5 libraries that are used, but it is worth trying.
    * If you are using Linux or Unix system, try to reach me on MS Teams (bd193399m@student.etf.bg.ac.rs), or email me on djordjebbozic@gmail.com.
    * Ask a colleague for help.