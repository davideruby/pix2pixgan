# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
import os
from tqdm import tqdm
import numpy as np
import albumentations
import torchvision
from albumentations.pytorch import ToTensorV2
import wandb
import config
import torch
import torch.nn as nn
from dataset.pannuke import CancerInstanceDataset, denormalize
from dataset.unitopatho_mask import UTP_Masks
from discriminator_model import Discriminator
from generator_model import Generator
import torch.optim as optim
import utils
import pandas as pd
import torch.multiprocessing as mp
import torch.distributed as dist


def main(gpu):
    # DDP
    print(f"#{gpu} started", flush=True)
    world_size = config.NGPU * config.NUM_NODES
    nr = 0  # it is the rank of the current node. Now we use only one node
    rank = nr * config.NGPU + gpu
    setup_ddp(rank, world_size)
    torch.cuda.set_device(gpu)
    is_master = rank == 0  # only master logs

    if config.LOG_WANDB and is_master:
        init_wandb()

    utils.set_seed(config.SEED)
    num_classes = len(CancerInstanceDataset.labels())
    disc = Discriminator(in_channels=3 + num_classes).cuda(gpu)
    gen = Generator(in_channels=num_classes, features=64).cuda(gpu)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(config.ADAM_BETA1, config.ADAM_BETA2))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(config.ADAM_BETA1, config.ADAM_BETA2))
    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # DDP
    disc = nn.parallel.DistributedDataParallel(disc, device_ids=[gpu], broadcast_buffers=False)
    gen = nn.parallel.DistributedDataParallel(gen, device_ids=[gpu], broadcast_buffers=False)

    # weight initalization
    disc.apply(utils.init_weights)
    gen.apply(utils.init_weights)

    # load dataset
    train_loader, test_loader = load_dataset_UTP(rank, world_size)

    # GradScaler
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    if config.LOG_WANDB and is_master:
        test_batch = next(iter(test_loader))
        test_batch_im, test_batch_masks = test_batch
        img_masks_test = [CancerInstanceDataset.get_img_mask(mask).permute(2, 0, 1) for mask in test_batch_masks]
        wandb.log({"Real": wandb.Image(torchvision.utils.make_grid(test_batch_im)),
                   "Masks": wandb.Image(torchvision.utils.make_grid(img_masks_test))})

    # training loop
    for epoch in range(config.NUM_EPOCHS):
        g_loss, d_loss = train_fn(disc, gen, train_loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, gpu)

        if config.SAVE_MODEL and epoch % 5 == 0 and is_master:
            utils.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            utils.save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        if config.LOG_WANDB and is_master:
            gen.eval()
            with torch.no_grad():
                fakes = gen(test_batch_masks.cuda(gpu))
                wandb.log({"generator_loss": g_loss, "discriminator_loss": d_loss, "Fakes": wandb.Image(fakes)})
            gen.train()
        # break

    # save gen and disc models
    if is_master:
        utils.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
        utils.save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

    if config.LOG_WANDB and is_master:
        wandb_log_generated_images(gen, test_loader)
        wandb.finish()

    torch.distributed.barrier()
    dist.destroy_process_group()


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = os.environ.get("MASTER_PORT", "24129")
    dist.init_process_group(
        backend='nccl',
        # init_method='env://',
        world_size=world_size,
        rank=rank
    )


def init_wandb():
    # my W&B (Rubinetti)
    # wandb.login(key="58214c04801c157c99c68d2982affc49dd6e4072")

    # EIDOSLAB W&B
    wandb.login(host='https://wandb.opendeephealth.di.unito.it',
                key='local-1390efeac4c23e0c7c9c0fad95f92d3c8345c606')
    wandb.init(project="unitopatho-generative",
               config={
                   "seed": config.SEED,
                   "device": config.DEVICE,
                   "root": config.ROOT_DIR,
                   "epochs": config.NUM_EPOCHS,
                   "lr": config.LEARNING_RATE,
                   "num_workers": config.NUM_WORKERS,
                   "l1_lambda": config.L1_LAMBDA,
                   "adam_beta1": config.ADAM_BETA1,
                   "adam_beta2": config.ADAM_BETA2,
                   "batch_size": config.BATCH_SIZE,
                   "virtual_batch_size": config.VIRTUAL_BATCH_SIZE,
                   "smooth_positive_labels": config.SMOOTH_POSITIVE_LABELS,
                   "smooth_negative_labels": config.SMOOTH_NEGATIVE_LABELS
               })
    config.CHECKPOINT_GEN = os.path.join(wandb.run.dir, config.CHECKPOINT_GEN)
    config.CHECKPOINT_DISC = os.path.join(wandb.run.dir, config.CHECKPOINT_DISC)


def wandb_log_generated_images(gen, loader, batch_to_log=5):
    images_to_log = []
    gen.eval()

    for idx_batch, (images_real, masks) in enumerate(loader):
        images_real, masks = images_real.to(config.DEVICE), masks.to(config.DEVICE)

        with torch.no_grad():
            fakes = gen(masks)

            for idx_sample, fake_img in enumerate(fakes):  # for each sample in batch
                real = denormalize(images_real[idx_sample]).cpu()
                mask = CancerInstanceDataset.get_img_mask(masks[idx_sample].cpu()).permute(2, 0, 1).cpu()
                fake = denormalize(fake_img).cpu()
                images_to_log.append(torchvision.utils.make_grid([mask, real, fake]))

        if idx_batch + 1 == batch_to_log:
            break

    wandb.log({"Generated_images (mask-real-fake)": [wandb.Image(img) for img in images_to_log]})
    gen.train()


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, gpu):
    loop = tqdm(loader, leave=True)
    do_log = gpu == 0
    patch_size = stride = 256
    disc_losses = []
    gen_losses = []
    gen.train()
    disc.train()

    for idx, (image_real, mask) in enumerate(loop):
        image_real = image_real.cuda(gpu)
        mask = mask.cuda(gpu)
        # le immagini sono 2048x2048, con patch_size = 256 vengono 8*8 patch
        image_real = get_patches(image_real, patch_size, stride)  #
        mask = get_patches(mask, patch_size, stride)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            image_fake = gen(mask)
            # real batch
            d_real = disc(mask, image_real)
            target = torch.ones_like(d_real)
            if config.SMOOTH_POSITIVE_LABELS:
                target = utils.smooth_positive_labels(target)
            d_real_loss = bce(d_real, target)
            # fake batch
            d_fake = disc(mask, image_fake.detach())
            target = torch.zeros_like(d_fake)
            if config.SMOOTH_NEGATIVE_LABELS:
                target = utils.smooth_negative_labels(target)
            d_fake_loss = bce(d_fake, target)
            # the paper says: "we divide the objective by 2 while optimizing D, which slows down the rate at
            # which D learns relative to G"
            d_loss = (d_real_loss + d_fake_loss) / 2
            disc_losses.append(d_loss.item())

            d_loss /= config.VIRTUAL_BATCH_SIZE

        # backward with gradient accumulation
        d_scaler.scale(d_loss).backward()
        if (idx + 1) % config.VIRTUAL_BATCH_SIZE == 0:
            d_scaler.step(opt_disc)
            d_scaler.update()
            opt_disc.zero_grad()

        # Train generator
        with torch.cuda.amp.autocast():
            d_fake = disc(mask, image_fake)
            g_fake_loss = bce(d_fake, torch.ones_like(d_fake))
            l1 = l1_loss(image_fake, image_real) * config.L1_LAMBDA
            g_loss = g_fake_loss + l1
            gen_losses.append(g_loss.item())

            g_loss /= config.VIRTUAL_BATCH_SIZE

        # backward with gradient accumulation
        g_scaler.scale(g_loss).backward()
        if (idx + 1) % config.VIRTUAL_BATCH_SIZE == 0:
            g_scaler.step(opt_gen)
            g_scaler.update()
            opt_gen.zero_grad()

        if idx % 10 == 0 and do_log:
            loop.set_postfix(D_real=torch.sigmoid(d_real).mean().item(), D_fake=torch.sigmoid(d_fake).mean().item())
            # print(f"gpu {gpu} has arrived at {idx + 1} / {len(loader)}", flush=True)
        # break
    return np.mean(gen_losses), np.mean(disc_losses)


def load_dataset_UTP(rank, world_size):
    path = '../data/unitopath-public/800'
    path_masks = "../data/unitopath-public/generated"

    # training set
    transform_train = albumentations.Compose([
        albumentations.Resize(height=2048, width=2048),
        albumentations.Flip(p=0.75),
        albumentations.RandomRotate90(p=0.75),
        albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    df = pd.read_csv(os.path.join(path, 'train.csv'))
    df = df[df.grade >= 0].copy()
    train_dataset = UTP_Masks(df, T=transform_train, path=path, target='grade', path_masks=path_masks, train=True)
    # DDP
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=config.BATCH_SIZE,
                                               pin_memory=True, num_workers=config.NUM_WORKERS, sampler=train_sampler)

    # test set
    transform_test = albumentations.Compose([
        albumentations.Resize(height=2048, width=2048),
        albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    df = pd.read_csv(os.path.join(path, 'test.csv'))
    df = df[df.grade >= 0].copy()
    test_dataset = UTP_Masks(df, T=transform_test, path=path, target='grade', path_masks=path_masks, train=False)
    # DDP
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=config.BATCH_SIZE,
                                              pin_memory=True, num_workers=config.NUM_WORKERS, sampler=test_sampler)

    return train_loader, test_loader


def get_patches(batch, kernel_size, stride):
    b, c, h, w = batch.shape
    # per dividere immagine originale in patches
    patches = batch.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)  # (b, c, h/k, w/k, k, k)  k=kernel_size
    patches = patches.reshape(b, c, -1, kernel_size, kernel_size)  # dim: (b, c, num_patches, k, k)
    patches = patches.permute(0, 2, 1, 3, 4)  # dim: (batch_size, num_patches, c, k, k)
    patches = patches.reshape(-1, c, kernel_size, kernel_size)  # dim: (b * num_patches, c, k, k)
    return patches


if __name__ == "__main__":
    print(f"Working on {config.DEVICE} device.")
    if "cuda" in str(config.DEVICE):
        for idx in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(idx))

    # DistributedDataParallel
    mp.spawn(main, nprocs=config.NGPU, args=())
