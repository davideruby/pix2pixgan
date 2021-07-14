import os
import torch
import utils
import torch.nn as nn
import torch.optim as optim
import torchvision
import config
from dataset.pannuke import CancerInstanceDataset, denormalize, visualize
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import wandb

torch.backends.cudnn.benchmark = True


def main():
    num_classes = len(CancerInstanceDataset.labels())
    disc = Discriminator(in_channels=3 + num_classes).to(config.DEVICE)
    gen = Generator(in_channels=num_classes, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(config.ADAM_BETA1, config.ADAM_BETA2))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(config.ADAM_BETA1, config.ADAM_BETA2))
    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # weight initalization
    disc.apply(utils.init_weights)
    gen.apply(utils.init_weights)

    if config.LOAD_MODEL:
        utils.load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        utils.load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    # train loader
    train_dataset = CancerInstanceDataset(train=True, transform=config.transform_training, download=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    # test loader
    test_dataset = CancerInstanceDataset(train=False, transform=config.transform_test, download=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # GradScaler
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # log wandb
    if config.LOG_WANDB:
        test_batch = next(iter(test_loader))
        test_batch_im, test_batch_masks = test_batch
        img_masks_test = [CancerInstanceDataset.get_img_mask(mask).permute(2, 0, 1) for mask in test_batch_masks]
        wandb.log({"Real": wandb.Image(torchvision.utils.make_grid(test_batch_im)),
                   "Masks": wandb.Image(torchvision.utils.make_grid(img_masks_test))})

    # training loop
    for epoch in range(config.NUM_EPOCHS):
        g_loss, d_loss = train_fn(disc, gen, train_loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler)

        if config.SAVE_MODEL and epoch % 5 == 0:
            utils.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            utils.save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        # save_some_examples(gen, test_loader, epoch, folder="evaluation")

        if config.LOG_WANDB and epoch % 5 == 0:
            gen.eval()
            with torch.no_grad():
                fakes = gen(test_batch_masks.to(config.DEVICE))
                wandb.log({"generator_loss": g_loss,
                           "discriminator_loss": d_loss,
                           "Fakes": wandb.Image(torchvision.utils.make_grid(fakes))})
            gen.train()

    # save gen and disc models
    utils.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
    utils.save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

    # show_generated_images(gen, test_loader)

    if config.LOG_WANDB:
        wandb_log_generated_images(gen, test_loader)
        wandb.finish()


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)
    disc_losses = []
    gen_losses = []
    gen.train()
    disc.train()

    for idx, (image_real, mask) in enumerate(loop):
        image_real = image_real.to(config.DEVICE)
        mask = mask.to(config.DEVICE)

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

        opt_disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            d_fake = disc(mask, image_fake)
            g_fake_loss = bce(d_fake, torch.ones_like(d_fake))
            l1 = l1_loss(image_fake, image_real) * config.L1_LAMBDA
            g_loss = g_fake_loss + l1
            gen_losses.append(g_loss.item())

        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(d_real).mean().item(),
                D_fake=torch.sigmoid(d_fake).mean().item(),
            )

    return np.mean(gen_losses), np.mean(disc_losses)


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


def show_generated_images(gen, loader):
    batch = next(iter(loader))
    images, masks = batch
    images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)

    gen.eval()
    with torch.no_grad():
        fakes = gen(masks)

        images = denormalize(images).cpu()
        masks = masks.cpu()
        fakes = denormalize(fakes).cpu()

        visualize(
            Masks=[CancerInstanceDataset.get_img_mask(mask).permute(2, 0, 1) for mask in masks],
            Real_Images=images,
            Fakes_Images=fakes
        )


if __name__ == "__main__":
    if config.LOG_WANDB:
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
                       "smooth_positive_labels": config.SMOOTH_POSITIVE_LABELS,
                       "smooth_negative_labels": config.SMOOTH_NEGATIVE_LABELS
                   })
        config.CHECKPOINT_GEN = os.path.join(wandb.run.dir, config.CHECKPOINT_GEN)
        config.CHECKPOINT_DISC = os.path.join(wandb.run.dir, config.CHECKPOINT_DISC)

    print(f"Working on {config.DEVICE} device.")
    if config.DEVICE == "cuda":
        for idx in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(idx))
    main()
