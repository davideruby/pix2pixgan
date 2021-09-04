# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
import math
import os
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm
import config
import utils
from dataset.pannuke import CancerInstanceDataset, denormalize
from dataset.unitopatho_mask import UTP_Masks
from discriminator_model import Discriminator
from generator_model import Generator


def main(gpu):
    # DDP
    print(f"GPU #{gpu} started")
    world_size = config.NGPU * config.NUM_NODES
    nr = 0  # it is the rank of the current node. Now we use only one node
    rank = nr * config.NGPU + gpu
    setup_ddp(rank, world_size)
    torch.cuda.set_device(gpu)
    is_master = rank == 0
    do_log = config.LOG_WANDB and is_master  # only master logs
    if do_log:
        wandb_init()

    # Load models
    num_classes = len(CancerInstanceDataset.labels())
    disc = Discriminator(in_channels=3 + num_classes).cuda(gpu)
    gen = Generator(in_channels=num_classes, features=64).cuda(gpu)
    # Use SynchBatchNorm for Multi-GPU trainings
    disc = nn.SyncBatchNorm.convert_sync_batchnorm(disc)
    gen = nn.SyncBatchNorm.convert_sync_batchnorm(gen)
    if do_log:
        print(disc)
        print(gen)

    # DDP
    disc = nn.parallel.DistributedDataParallel(disc, device_ids=[gpu])
    gen = nn.parallel.DistributedDataParallel(gen, device_ids=[gpu])

    # Optimizers
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(config.ADAM_BETA1, config.ADAM_BETA2))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(config.ADAM_BETA1, config.ADAM_BETA2))

    # Losses
    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # Load checkpoints from wandb
    if config.LOAD_MODEL:
        api = wandb.Api()
        run = api.run("daviderubi/pix2pixgan/1l0hnnnn")  # upbeat-river-42
        run.file("disc.pth").download(replace=True)
        run.file("gen.pth").download(replace=True)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        utils.load_checkpoint("disc.pth", disc, opt_disc, config.LEARNING_RATE, map_location=map_location)
        utils.load_checkpoint("gen.pth", gen, opt_gen, config.LEARNING_RATE, map_location=map_location)

    # load dataset
    train_loader, test_loader = load_dataset_UTP(rank, world_size)

    # grad_scaler
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    if do_log:
        # Get batch from testloader. Every epoch we will log the generated images for this batch on wandb.
        test_batch_im, test_batch_masks = wandb_get_images_to_log(test_loader)
        img_masks_test = [CancerInstanceDataset.get_img_mask(mask).permute(2, 0, 1) for mask in test_batch_masks]
        wandb.log({"Reals": wandb.Image(torchvision.utils.make_grid(test_batch_im), caption="Reals"),
                   "Masks": wandb.Image(torchvision.utils.make_grid(img_masks_test), caption="Masks")})

    # training loop
    for epoch in range(config.NUM_EPOCHS):
        g_adv_loss, g_l1_loss, d_loss = train_epoch(disc, gen, train_loader, opt_disc, opt_gen, l1_loss, bce, g_scaler,
                                                    d_scaler, gpu)

        if config.SAVE_MODEL and (epoch + 1) % 10 == 0 and is_master:
            print(f"Saving checkpoint at epoch {epoch + 1}...")
            utils.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN, epoch=epoch + 1)
            utils.save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC, epoch=epoch + 1)
            if do_log:
                wandb.save(config.CHECKPOINT_GEN)
                wandb.save(config.CHECKPOINT_DISC)

        if do_log:
            # Log generated images after the training epoch.
            gen.eval()
            with torch.no_grad():
                fakes = gen(test_batch_masks.cuda(gpu))
                fakes = denormalize(fakes)
                wandb.log({"generator_adv_loss": g_adv_loss,
                           "generator_l1_loss": g_l1_loss,
                           "discriminator_loss": d_loss,
                           "Fakes": wandb.Image(fakes, caption="Fakes")})
            gen.train()

    # Save generator and discriminator models.
    if is_master:
        utils.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN, epoch=config.NUM_EPOCHS)
        utils.save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC, epoch=config.NUM_EPOCHS)

    if do_log:
        # Log on wandb some generated images.
        wandb_log_generated_images(gen, test_loader, batch_to_log=math.ceil(100 / config.BATCH_SIZE))
        wandb.finish()

    torch.distributed.barrier()
    dist.destroy_process_group()


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = os.environ.get("MASTER_PORT", "24129")
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)


def wandb_init():
    # my W&B (Rubinetti)
    wandb.login(key="58214c04801c157c99c68d2982affc49dd6e4072")

    # EIDOSLAB W&B
    # wandb.login(host='https://wandb.opendeephealth.di.unito.it',
    #             key='local-1390efeac4c23e0c7c9c0fad95f92d3c8345c606')
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
                   "he_norm": config.HE_NORM,
                   "smooth_positive_labels": config.SMOOTH_POSITIVE_LABELS,
                   "smooth_negative_labels": config.SMOOTH_NEGATIVE_LABELS
               })
    config.CHECKPOINT_GEN = os.path.join(wandb.run.dir, config.CHECKPOINT_GEN)
    config.CHECKPOINT_DISC = os.path.join(wandb.run.dir, config.CHECKPOINT_DISC)


def wandb_log_generated_images(gen, loader, batch_to_log=5):
    images_to_log = []
    gen.eval()

    with torch.no_grad():
        for idx_batch, sample in enumerate(loader):
            reals = sample["image"].to(config.DEVICE)
            masks = sample["mask"].to(config.DEVICE)
            fakes = gen(masks)

            for fake, real, mask in zip(fakes, reals, masks):  # for each element in batch
                mask = CancerInstanceDataset.get_img_mask(mask.cpu()).permute(2, 0, 1).cpu()
                real = denormalize(real).cpu()
                fake = denormalize(fake).cpu()
                images_to_log.append(torchvision.utils.make_grid([mask, real, fake]))

            if idx_batch + 1 == batch_to_log:
                break

    wandb.log({"Generated_images (mask-real-fake)": [wandb.Image(img, caption="Mask - Real - Fake") for img in
                                                     images_to_log]})
    gen.train()


def wandb_get_images_to_log(loader, num_img_to_log=10):
    """
    :param loader: loader of dataset.
    :param num_img_to_log: how many images you want to log on wandb.
    :return: num_img_to_log images and masks thought to be logged.
    """
    imgs = []
    masks = []
    count = 0

    for img, mask in loader:
        imgs.append(img)
        masks.append(mask)
        count += img.size()[0]
        if count >= num_img_to_log:
            break

    test_batch_im = torch.cat(imgs, dim=0)[:num_img_to_log]
    test_batch_mask = torch.cat(masks, dim=0)[:num_img_to_log]
    test_batch_im = denormalize(test_batch_im)
    return test_batch_im.cpu(), test_batch_mask.cpu()


def train_epoch(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, gpu):
    loop = tqdm(loader, leave=True)
    do_log = gpu == 0
    disc_losses = []
    gen_l1_losses = []
    gen_adv_losses = []
    gen.train()
    disc.train()
    five_crop = any(isinstance(tr, transforms.FiveCrop) for tr in loader.dataset.transform.transforms)

    for idx, sample in enumerate(loop):
        real_image = sample["image"].cuda(gpu)
        mask = sample["mask"].cuda(gpu)

        # fuse batch size and ncrops
        if five_crop:
            bs, ncrops, c_img, h, w = real_image.size()
            c_mask = mask.size()[2]
            real_image = real_image.view(-1, c_img, h, w)  # bs * ncrops, c, h, w
            mask = mask.view(-1, c_mask, h, w)  # bs * ncrops, c, h, w

        # Train Discriminator
        with torch.cuda.amp.autocast():
            fake_image = gen(mask)
            # real batch
            d_real = disc(mask, real_image)
            target = torch.ones_like(d_real)
            if config.SMOOTH_POSITIVE_LABELS:
                target = utils.smooth_positive_labels(target)
            d_real_loss = bce(d_real, target)
            # fake batch
            d_fake = disc(mask, fake_image.detach())
            target = torch.zeros_like(d_fake)
            if config.SMOOTH_NEGATIVE_LABELS:
                target = utils.smooth_negative_labels(target)
            d_fake_loss = bce(d_fake, target)
            # the paper says: "we divide the objective by 2 while optimizing D, which slows down the rate at
            # which D learns relative to G"
            d_loss = (d_real_loss + d_fake_loss) / 2
            disc_losses.append(d_loss.item())

        opt_disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            d_fake = disc(mask, fake_image)
            g_fake_loss = bce(d_fake, torch.ones_like(d_fake))
            l1 = l1_loss(fake_image, real_image) * config.L1_LAMBDA
            g_loss = g_fake_loss + l1
            gen_l1_losses.append(l1.item())
            gen_adv_losses.append(g_fake_loss.item())

        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0 and do_log:
            loop.set_postfix(D_real=torch.sigmoid(d_real).mean().item(), D_fake=torch.sigmoid(d_fake).mean().item())

    return np.mean(gen_adv_losses), np.mean(gen_l1_losses), np.mean(disc_losses)


def load_dataset_UTP(rank, world_size):
    path = '../data/unitopath-public/800'
    mask_dir = "generated_torchstain" if config.HE_NORM else "generated"
    path_masks = f"../data/unitopath-public/{mask_dir}"
    crop_size = 256

    # training set
    transform_train = transforms.Compose([
        transforms.FiveCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.RandomHorizontalFlip()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.RandomVerticalFlip()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([utils.RandomRotate90()(crop) for crop in crops])),
    ])
    df = pd.read_csv(os.path.join(path, 'train.csv'))
    df = df[df.grade >= 0].copy()
    train_dataset = UTP_Masks(df, T=transform_train, path=path, target='grade', path_masks=path_masks, train=True,
                              device=torch.cuda.current_device())
    # DDP
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=config.BATCH_SIZE,
                                               num_workers=config.NUM_WORKERS, sampler=train_sampler)

    transform_test = transforms.Compose([
        transforms.RandomCrop(1024),
    ])
    df = pd.read_csv(os.path.join(path, 'test.csv'))
    df = df[df.grade >= 0].copy()
    test_dataset = UTP_Masks(df, T=transform_test, path=path, target='grade', path_masks=path_masks, train=False,
                             device=torch.cuda.current_device())
    # DDP
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=config.BATCH_SIZE,
                                              num_workers=config.NUM_WORKERS, sampler=test_sampler)

    return train_loader, test_loader


if __name__ == "__main__":
    print(f"Working on {config.DEVICE} device.")
    if "cuda" in str(config.DEVICE):
        for idx in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(idx))

    # DistributedDataParallel
    mp.spawn(main, nprocs=config.NGPU, args=())
