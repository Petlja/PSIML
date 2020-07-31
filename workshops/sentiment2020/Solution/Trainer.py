from EvaluationHelper import compute_accuracy
from TwitterDataLoader import generate_batches
from TwitterDataset import TwitterDataset

class Trainer:
    def __init__(self, dataset, model, loss_func, optimizer):
        """
        Args:
            dataset (TwitterDataset): the dataset used for training and evaluation
            model (SentimentClassifierPerception): the model
            loss_func (torch.nn.CrossEntropyLoss): the loss function that should be used
            optimizer (torch.optim.Optimizer): the optimizer that should be used to update model weights
        """
        self.dataset = dataset
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer


    def train(self, num_epochs, batch_size, device):
        """
        Trains the model on the training set, for the chosen number of epochs.

        Args:
            num_epochs (int): the number of training iterations
            batch_size (int): the batch size that should be used
            device (str): the device where the data tensors should be stored: "cpu" or "gpu"
        Returns:
            report dictionary with all metrics measured during training (dict)
        """
        # initialize the metrics report
        report = self._initialize_report()

        for epoch_index in range(num_epochs):

            report["epoch_index"] = epoch_index

            # Iterate over training set to train the model
            running_loss, running_accuracy = self._train_epoch(batch_size, device)

            report["train_loss"].append(running_loss)
            report["train_accuracy"].append(running_accuracy)

            # Iterate over validation set to evaluate the model
            running_loss, running_accuracy = self.evaluate(batch_size, device)

            report["validation_loss"].append(running_loss)
            report["validation_accuracy"].append(running_accuracy)

        return report


    def evaluate(self, batch_size, device, split="validation"):
        """
        Evaluates model performance on the chosen dataset split.

        Args:
            batch_size (int): the batch size that should be used
            device (str): the device where the data tensors should be stored: "cpu" or "gpu"
            split (str): one of "train", "validation" or "test"
        Returns:
            average loss of the model (float), average accuracy of the model (float) on the training set
        """
        # set the dataset split
        self.dataset.set_split(split)
        # create a batch generator from the dataset
        batch_generator = generate_batches(self.dataset, batch_size=batch_size, device=device)

        # evaluate the model by tracking loss and accuracy for each batch
        running_loss, running_accuracy = self._evaluate_batches(batch_generator)

        return running_loss, running_accuracy


    def _initialize_report(self):
        """
        Initialize the dictionary containing all metrics measured during training

        Returns:
            Empty dictionary with all desired metrics set (dict)
        """ 
        report = {}
        report["epoch_index"] = []
        report["train_loss"] = []
        report["train_accuracy"] = []
        report["validation_loss"] = []
        report["validation_accuracy"] = []

        return report


    def _train_epoch(self, batch_size, device):
        """
        Loop that iterates over batches to train the model.
        The following 5 steps are repeated for each batch:
            1. Zero the gradients (clear the information about gradients from previous step)
            2. Calculate the model output
            3. Compute the loss, when compared with labels
            4. Use the loss to calculate and backpropagate gradients
            5. Use optimize to update weights of the model

        Args:
            batch_size (int): the batch size that should be used
            device (str): the device where the data tensors should be stored: "cpu" or "gpu"
        Returns:
            average loss of the model (float), average accuracy of the model (float) on the training set
        """
        # set the dataset split
        self.dataset.set_split("train")
        # create a batch generator from the dataset
        batch_generator = generate_batches(self.dataset, batch_size=batch_size, device=device)

        # initialize variables for tracking loss and accuracy in the batch
        running_loss = 0
        running_accuracy = 0

        # set the model in "training mode": the model parameters are mutable
        self.model.train()

        # iterate over training batches
        for batch_index, batch_dict in enumerate(batch_generator):

            # step 1: zero the gradients (clear the information about gradients from previous step)
            self.optimizer.zero_grad()

            # step 2: calculate the output
            y_pred = self.model(x_in=batch_dict["x_data"].float())

            # step 3: compute the loss
            loss = self.loss_func(y_pred, (batch_dict["y_target"]).long())
            
            # take only a value of the loss tensor
            loss_batch_value = loss.item()

            # update the "average loss" calculated for the batch
            running_loss += (loss_batch_value - running_loss) / (batch_index + 1)

            # step 4: use loss to calculate and backpropagate gradients
            loss.backward()

            # step 5: use optimizer to update weights
            self.optimizer.step()

            # bonus: compute accuracy
            accuracy_batch = compute_accuracy(y_pred, batch_dict["y_target"])

            # update the "average accuracy" calculated for the batch
            running_accuracy += (accuracy_batch - running_accuracy) / (batch_index + 1)

        return running_loss, running_accuracy


    def _evaluate_batches(self, batch_generator):
        """
        Loop that iterates over batches to evaluate the model performance.
        It has 3 steps, repeated for each batch:
            1. Calculate the model output
            2. Compute the loss, when compared with labels
            3. Compute the accuracy, when comapred with the labels

        Args:
            batch_generator (torch.DataLoader): generator through batches in the dataset
        Returns:
            average loss of the model (float), average accuracy of the model (float) on the dataset
        """
        # initialize variables for tracking loss and accuracy in the batch
        running_loss = 0
        running_accuracy = 0

        # set the model in "evaluation mode": the model parameters are immutable, loss is not calculated, gradients are not propagated
        self.model.eval()

        # iterate over batches
        for batch_index, batch_dict in enumerate(batch_generator):

            # step 1: compute the output
            y_pred = self.model(x_in=batch_dict["x_data"].float())

            # step 2: compute the loss
            loss = self.loss_func(y_pred, (batch_dict["y_target"]).long())
            
            # take only a value of the loss tensor
            loss_batch_value = loss.item()

            # update the "average loss" calculated for the batch
            running_loss += (loss_batch_value - running_loss) / (batch_index + 1)

            # step 3: compute accuracy
            accuracy_batch = compute_accuracy(y_pred, batch_dict["y_target"])

            # update the "average accuracy" calculated for the batch
            running_accuracy += (accuracy_batch - running_accuracy) / (batch_index + 1)

        return running_loss, running_accuracy
