import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from critics import FCCritic
from generators import FCGenerator
from dataset import FacesDataSet

IMG_SIZE = 128 # Images will be IMG_SIZExIMG_SIZE
CHANNELS = 3 # RGB
Z_SIZE = 100 # Size of latent vector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Boilerplate code for using CUDA for faster training
CUDA = torch.cuda.is_available() # Use CUDA for faster training
MAX_SUMMARY_IMAGES = 4 # How many images to output to Tensorboard
LR = 1e-4 # Learning rate
EPOCHS = 20 # Number of epochs
BATCH_SIZE = 64 # Number of images in a batch
NUM_WORKERS = 8  # How many parallel workers for data ingestion. NOTE: Set to 0 when debugging
CLIP_VALUE = 1e-2 # Used to clip the parameters of the critic network

assert MAX_SUMMARY_IMAGES <= BATCH_SIZE, "We can only write to Tensorboard as many images as there are in a batch"

def train():
    # Initialize the Tensorboard summary. Logs will end up in runs directory
    summary_writer = SummaryWriter()

    # Initialize the critic and the generator. NOTE: You can use other classes from critis.py and generators.py here.
    critic = FCCritic(IMG_SIZE, CHANNELS)
    generator = FCGenerator(IMG_SIZE, CHANNELS, Z_SIZE)

    critic.to(DEVICE)
    generator.to(DEVICE)

    # Initialize the data set. NOTE: You can pass total_images argument to avoid loading the whole dataset.
    data_set = FacesDataSet(IMG_SIZE)
    total_iterations = len(data_set) // BATCH_SIZE
    data_loader = DataLoader(data_set,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=NUM_WORKERS)

    optimizer_c = torch.optim.RMSprop(critic.parameters(), lr=LR)
    optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        if epoch < 10:
            c_times = 100
        else:
            c_times = 5

        for i, real_img_batch in tqdm(enumerate(data_loader), total=total_iterations, desc=f"Epoch: {epoch}", unit="batches"):
            # Global step is used in tensorboard so that we have unique number for each batch of data
            global_step = epoch * total_iterations + i
            """
            ##################################################################
            TODO: Generate a batch of fake images.
            Hint: You'll first need a z_batch vector from which you'll generate images.
            NOTE: You DO NOT want to update the generator's parameters at this point.
            YOUR CODE BEGIN.
            ##################################################################
            """
            z_batch = None
            # This is just boilerplate if you're using CUDA - All inputs to the network need to be on the same device
            real_img_batch = real_img_batch.to(DEVICE)
            z_batch = z_batch.to(DEVICE)

            fake_img_batch = None

            """
            ##################################################################
            YOUR CODE END.
            ##################################################################
            """

            optimizer_c.zero_grad()

            """
            ##################################################################
            TODO: Implement adverserial loss for the critic.
            Hint: You'll need to know the 'score' on both the real and the fake image batch
            YOUR CODE BEGIN.
            ##################################################################
            """

            loss_c = None

            """
            ##################################################################
            YOUR CODE END.
            ##################################################################
            """

            summary_writer.add_scalar("Critic loss", loss_c, global_step)
            # Calculate the gradients with respect to the input
            loss_c.backward()
            # Apply backward prop
            optimizer_c.step()
            # Clip the weights of the critic to satisfy the Lipschitz constraint
            for p in critic.parameters():
                p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

            # Train the generator only after the critic has been trained c_times
            if i % c_times == 0:
                optimizer_g.zero_grad()

                """
                ##################################################################
                TODO: Implement adverserial loss for the generator.
                Hint: You'll need to know the 'score' on a new batch of generated images.

                YOUR CODE BEGIN.
                ##################################################################
                """
                gen_imgs = None
                # Adversarial loss
                loss_g = None
                summary_writer.add_scalar("Generator loss", loss_g, global_step)

                # Calculate the gradients with respect to the input
                loss_g.backward()
                # Apply backward prop
                optimizer_g.step()

                summary_writer.add_images("Generated images", gen_imgs[:MAX_SUMMARY_IMAGES], global_step)

if __name__ == "__main__":
    train()