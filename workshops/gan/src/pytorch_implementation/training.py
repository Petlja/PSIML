import torch
from torch.utils.tensorboard import SummaryWriter  # TODO: use it
from torch.utils.data import DataLoader
from tqdm import tqdm

from critics import FCCritic
from generators import FCGenerator
from dataset import FacesDataSet

IMG_SIZE = 128
CHANNELS = 3
Z_SIZE = 100

CUDA = True if torch.cuda.is_available() else False
LR = 1e-4
EPOCHS = 1
BATCH_SIZE = 64
NUM_WORKERS = 8  # set to 0 when debugging
CLIP_VALUE = 1e-2

def train():
    critic = FCCritic(IMG_SIZE, CHANNELS)
    generator = FCGenerator(IMG_SIZE, CHANNELS, Z_SIZE)
    if CUDA:
        critic.cuda()
        generator.cuda()

    data_loader = DataLoader(FacesDataSet(IMG_SIZE),
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=NUM_WORKERS)

    optimizer_C = torch.optim.RMSprop(critic.parameters(), lr=LR)
    optimizer_G = torch.optim.RMSprop(critic.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        if epoch < 25:
            c_times = 100
        else:
            c_times = 5

        for i, real_img_batch in tqdm(enumerate(data_loader)):
            z_batch = torch.rand(BATCH_SIZE, Z_SIZE)
            if CUDA:
                real_img_batch = real_img_batch.cuda()
                z_batch = z_batch.cuda()

            optimizer_C.zero_grad()
            fake_img_batch = generator(z_batch).detach()
            # Adversarial loss
            loss_c = -torch.mean(critic(real_img_batch)) + torch.mean(critic(fake_img_batch))

            # Calculate the gradients with respect to the input
            loss_c.backward()
            # Apply backward prop
            optimizer_C.step()

            # Clip the weights of the discriminator to satisfy the Lipschitz constraint
            for p in critic.parameters():
                p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

            # Train the generator only after the discriminator has been trained c_times
            if i % c_times == 0:
                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(z_batch)
                # Adversarial loss
                loss_G = -torch.mean(critic(gen_imgs))

                loss_G.backward()
                optimizer_G.step()

if __name__ == "__main__":
    train()