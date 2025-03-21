import os
import yaml
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image

from models import Generator, Discriminator
from utils import initialize_weights

def train_network(cfg: str):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and cfg['gpu'] > 0 else "cpu")

    # Initialize models
    netG = Generator(cfg['input_nc'], cfg['output_nc'], cfg['ngf']).to(device)
    netD = Discriminator(cfg['input_nc'], cfg['output_nc'], cfg['ndf']).to(device)

    # Initialize weights
    netG.apply(initialize_weights)
    netD.apply(initialize_weights)

    # Loss functions
    criterion = nn.BCELoss().to(device)
    criterionAE = nn.L1Loss().to(device)

    # Optimizers
    cfg_opt = cfg['optimizer']
    optimizerG = optim.Adam(netG.parameters(), lr=cfg_opt['lr'], betas=(cfg_opt['beta1'], 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=cfg_opt['lr'], betas=(cfg_opt['beta1'], 0.999))

    # Data loader
    # transform = transforms.Compose([
    #     transforms.Resize(cfg['image-size']['loadSize']),
    #     transforms.RandomCrop(cfg['image-size']['fineSize']),
    #     transforms.RandomHorizontalFlip() if cfg['augment']['flip'] else transforms.Lambda(lambda x: x),
    #     transforms.ToTensor()
    # ])

    dataset = datasets.ImageFolder(cfg['data_path'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg['batchSize'], num_workers=cfg['nThreads'])

    # Training loop
    print_freq = cfg['saving']['print_freq']
    epochs = cfg['optimizer']['epochs']
    save_epoch_freq = cfg['saving']['save_epoch_freq']

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            real_A, _ = data
            real_B = real_A.clone()  # Assuming paired dataset

            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Update Discriminator
            netD.zero_grad()
            output = netD(real_A)
            label = torch.full(output.size(), 1, dtype=torch.float, device=device)
            errD_real = criterion(output, label)
            errD_real.backward()

            fake_B = netG(real_A)
            output = netD(fake_B.detach())
            label.fill_(0)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()

            # Update Generator
            netG.zero_grad()
            output = netD(fake_B)
            label.fill_(1)
            errG = criterion(output, label)
            errL1 = criterionAE(fake_B, real_B) * cfg['loss']['lambda_L1']
            errG_total = errG + errL1
            errG_total.backward()
            optimizerG.step()

            if i % print_freq == 0:
                print(f'Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] '
                      f'Loss_D: {errD_real.item() + errD_fake.item()} Loss_G: {errG.item()} Loss_L1: {errL1.item()}')

        if epoch % save_epoch_freq == 0:
            torch.save(netG.state_dict(), os.path.join(cfg['saving']['checkpoints_dir'], f'{epoch}_net_G.pth'))
            torch.save(netD.state_dict(), os.path.join(cfg['saving']['checkpoints_dir'], f'{epoch}_net_D.pth'))

        print(f'End of epoch {epoch}/{epochs}')

    # Save final models
    torch.save(netG.state_dict(), os.path.join(cfg['saving']['checkpoints_dir'], 'final_net_G.pth'))
    torch.save(netD.state_dict(), os.path.join(cfg['saving']['checkpoints_dir'], 'final_net_D.pth'))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = 'config.yaml'
    with open(path, 'r') as file:
        cfg = yaml.safe_load(file)

    torch.manual_seed(42)
    train_network(cfg)