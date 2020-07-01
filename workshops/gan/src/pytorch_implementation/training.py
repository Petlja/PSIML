import torch
import torchvision
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
MAX_SUMMARY_IMAGES = 4
LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 64
NUM_WORKERS = 8  # set to 0 when debugging
CLIP_VALUE = 1e-2

assert MAX_SUMMARY_IMAGES <= BATCH_SIZE

def train():
    summary_writer = SummaryWriter()

    critic = FCCritic(IMG_SIZE, CHANNELS)
    generator = FCGenerator(IMG_SIZE, CHANNELS, Z_SIZE)
    if CUDA:
        critic.cuda()
        generator.cuda()

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
            global_step = epoch * total_iterations + i
            z_batch = torch.rand(BATCH_SIZE, Z_SIZE)
            if CUDA:
                real_img_batch = real_img_batch.cuda()
                z_batch = z_batch.cuda()

            optimizer_c.zero_grad()
            fake_img_batch = generator(z_batch).detach()
            # Adversarial loss
            loss_c = -torch.mean(critic(real_img_batch)) + torch.mean(critic(fake_img_batch))

            summary_writer.add_scalar("Critic loss", loss_c, global_step)
            # Calculate the gradients with respect to the input
            loss_c.backward()
            # Apply backward prop
            optimizer_c.step()

            # Clip the weights of the discriminator to satisfy the Lipschitz constraint
            for p in critic.parameters():
                p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

            # Train the generator only after the discriminator has been trained c_times
            if i % c_times == 0:
                optimizer_g.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(z_batch)
                # Adversarial loss
                loss_g = -torch.mean(critic(gen_imgs))
                summary_writer.add_scalar("Generator loss", loss_g, global_step)

                loss_g.backward()
                optimizer_g.step()

                summary_writer.add_images("Generated images", gen_imgs[:MAX_SUMMARY_IMAGES], global_step)

if __name__ == "__main__":
    train()