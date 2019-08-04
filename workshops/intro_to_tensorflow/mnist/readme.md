# MNIST Workshop
This is the second of two workshops designed to cover basic use of low level TensorFlow APIs, building and training simple neural network.

So far, we saw TensorFlow in use, and played with neural network architectures and training.

Next step is to adjust the pipeline from the previous workshop to a new dataset - MNIST Fashion dataset.

## Main goal
You should change the model's architecture, as well as Trainer class, to work with the new data. You will also see how last example applies to the real data application. You will work with tf.Session.

## Workshop guide
Same as before, there are four main parts of the workshop:
* **Dataset**: load - visualize
* **Model**: create tf.Graph with network
* **Training**: train Model on dataset - accuracy
* **TensorBoard**: graph - loss - accuracy

### Dataset
All of dataset related logic is in `data.py` file.
You don't *need* to look into the code.

Dataset is loaded by calling `get_fashion_mnist()`.

Dataset is made up of 28x28 images of clothing items. Your goal is to classify those images. Each image is labeled with one of 10 classes:
* T-shirt/top
* Trouser
* Pullover
* Dress
* Coat
* Sandal
* Shirt
* Sneaker
* Bag
* Ankle boot

![MNIST Fashion](resources/mnist.png)

### Model
There are few things you need to implement/adjust in `model.py`:
* Input placeholder
* Flat features
* New layers
* Output layer for 10 classes
* Training operation

As you're finding your way through the code, you will see more tips that will guide your implementation.

### Training
Once you change the network, you need to adjust `Trainer` in `trainer.py` to feed new data into new placeholders. Consult with the previous workshop and add running tf.Session in both train epoch and valid epoch.

### TensorBoard
Make sure you restart the TensorBoard, now in the *mnist/output* folder:

```
tensorboard --logdir *path_root*/mnist/output
```

You will again be able to see Graph, loss and accuracy.

## So what should you do?

1. Go and implement all TODOs in `model.py` and `trainer.py`
2. Experiment with network layers
3. Edit training parameters if needed in `run.py`
4. See results in [TensorBoard](#TensorBoard)
5. Repeat 1-4 until you have accuracy over 0.9, and continue to get the best you can

### If you still have time

6. Add saving accuracy per class
7. Visualize images that are wrongly classified
