# Generative Adversarial Networks Lecture

This repository contains the presentation and the code for the workshop for the GAN lecture on PSIML 2018.

![](workshops/gan/assets/gan_training_regime.png)

---

### Presentation

Presentation slides can be found [here](https://github.com/bgavran/GAN_Lecture_Materials/blob/master/presentation/presentation.pdf).

### Workshop starting point

You will be training your own Wasserstein GAN with Gradient Penalty on the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset.

The code uses Python 3.6.5 and TensorFlow 1.9.0 version (but TensorFlow >= 1.0 should be fine).

The data loading pipeline and most of the boilerplate code is already written.

Your task is to download the dataset and fill in the missing parts of the code.

You should be able to train a rudimentary WGAN-GP with fully connected generator and critic on a low resolution Faces dataset in a matter of minutes, even on the CPU of a low-end PC.

---

#### Step 1. - Download the dataset
The recommended version is the [deep funneled version](http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz). 
The code will look for images recursively in the `gan/data/lfw-deepfunnelled` directory.

#### Step 2. - Fill in the missing parts of the code!

Starting point is the `main.py` file, which sets up and trains your model, but the only file you need to edit is `code/src/wgan.py`.

There are three places where you need to put your code: 
* Cost functions for the critic and the generator
* Optimizers for the critic and the generator
* Training regime


Network architectures for the generators and the critics are already provided. 
Place for your code is marked in the file. 
The total amount of code isn't huge - it should be less than 50 lines.

Although you don't need to touch anything else to make it work, feel free to experiment and adjust the code to your liking!

#### Step 3. - Train your model!

Run the main.py and check your model performance in TensorBoard!

Run the `tensorboard --logdir=log --reload_interval=1` from your `code/` directory and access the `localhost:6006` address from your browser.
You'll be able to track values of losses, generated images as well as the distribution of weights and gradients of all the parameters in your network.

For reference, `Intel i5-4200U CPU @ 1.60GHz` trains the basic shape of the face within first 200 iterations!

#### Step 4.
???

#### Step 5.
Profit!!!

Feel free to expriment with various architectures for generator and the critic! 

Download some other dataset! 

Try removing the Lipschitz regularization and see how the model fails!

Try to figure out your own way to enforce the Lipschitz constraint! (and publish a paper about it!)

If you have access to a GPU farm, try generating HD images!
